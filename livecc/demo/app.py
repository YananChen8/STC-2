import argparse

parser = argparse.ArgumentParser(description="Set runtime flags")
parser.add_argument("--hf_spaces", action="store_true", help="Use this flag if running on Hugging Face Spaces.")
parser.add_argument("--js_monitor", action="store_true", default=True,
                    help="Whether to use JS-based video timestamp monitoring (disable for environments with high latency).")
parser.add_argument("--model-path", default=None, help="HF model id, snapshot dir, or HF cache repo dir.")
parser.add_argument("--device", default=None, help="cuda, cpu, mps, or cuda:0 style device.")
parser.add_argument("--share", action="store_true", help="Enable Gradio share tunnel.")

args = parser.parse_args()

hf_spaces = args.hf_spaces
js_monitor = args.js_monitor

if hf_spaces:
    try:
        import spaces
    except Exception as e:
        print(e)

import os
import numpy as np
import gradio as gr

try:
    from demo.infer import LiveCCDemoInfer
except ModuleNotFoundError:
    from infer import LiveCCDemoInfer

class GradioBackend:
    waiting_video_response = 'Waiting for video input...'
    not_found_video_response = 'Video does not exist...'
    mode2api = {
        'Real-Time Commentary': 'live_cc',
        'Conversation': 'video_qa'
    }
    def __init__(self, model_path: str = None, device: str = None):
        self.infer = LiveCCDemoInfer(model_path=model_path, device=device)
    
    def __call__(self, message: str = None, history: list[str] = None, state: dict = {}, mode: str = 'Real-Time Commentary', **kwargs):
        return getattr(self.infer, self.mode2api[mode])(message=message, history=history, state=state, **kwargs)

gradio_backend = None if hf_spaces else GradioBackend(model_path=args.model_path, device=args.device)

with gr.Blocks() as demo:
    gr.Markdown("## LiveCC Conversation and Real-Time Commentary - Gradio Demo")
    gr.Markdown("### [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale (CVPR 2025)](https://showlab.github.io/livecc/)")
    gr.Markdown("1️⃣ Select Mode, Real-Time Commentary (LiveCC) or Conversation (Common QA/Multi-turn)")
    gr.Markdown("2️⃣🅰️ **Real-Time Commentary:  Input a query (optional) -> Click or upload a video**.")
    gr.Markdown("2️⃣🅱️ **Conversation: Click or upload a video -> Input a query**.")
    # gr.Markdown("*HF Space Gradio has unsolvable latency (10s~20s), and not support flash-attn. If you want to enjoy the very real-time experience, please deploy locally https://github.com/showlab/livecc*")
    gr_state = gr.State({}, render=False) # control all useful state, including kv cache
    gr_video_state = gr.JSON({}, visible=False) # only record video state, belong to gr_state but lightweight
    gr_static_trigger = gr.Number(value=0, visible=False) # control start streaming or stop
    gr_dynamic_trigger = gr.Number(value=0, visible=False) # for continuous refresh 
    
    with gr.Row():
        with gr.Column():
            gr_video = gr.Video(
                label="video",
                elem_id="gr_video",
                visible=True,
                sources=['upload'],
                autoplay=True,
                width=720,
                height=480
            )
            gr_examples = gr.Examples(
                examples=[
                    'demo/sources/howto_fix_laptop_mute_1080p.mp4',
                    'demo/sources/warriors_vs_rockets_2025wcr1_mute_1080p.mp4',
                    'demo/sources/cvpr25_vlog.mp4',
                ],
                inputs=[gr_video],
            )
            gr_clean_button = gr.Button("Clean (Press me before changing video)", elem_id="gr_button")

        with gr.Column():
            with gr.Row():
                gr_radio_mode = gr.Radio(label="Select Mode", choices=["Real-Time Commentary", "Conversation"], elem_id="gr_radio_mode", value='Real-Time Commentary', interactive=True) 

            # @spaces.GPU
            def gr_chatinterface_fn(message, history, state, video_path, mode):
                if mode != 'Conversation':
                    yield 'waiting for video input...', state
                    return
                global gradio_backend
                if gradio_backend is None:
                    yield '(ZeroGPU needs to initialize model under @spaces.GPU, thanks for waiting...)', state
                    gradio_backend = GradioBackend(model_path=args.model_path, device=args.device)
                    yield '(finished initialization, responding...)', state
                state['video_path'] = video_path
                response, state = gradio_backend(message=message, history=history, state=state, mode=mode, hf_spaces=hf_spaces)
                yield response, state
                
            def gr_chatinterface_chatbot_clear_fn(gr_dynamic_trigger):
                gradio_backend.infer._cached_video_readers_with_hw = {}
                return {}, {}, 0, gr_dynamic_trigger
            gr_chatinterface = gr.ChatInterface(
                fn=gr_chatinterface_fn,
                type="messages", 
                additional_inputs=[gr_state, gr_video, gr_radio_mode],
                additional_outputs=[gr_state],
            )
            gr.Examples(
                examples=["Please commentate on Game 1 of 2025 Playoffs First Round.", "My VLog~"],
                inputs=[gr_chatinterface.textbox]
            )
            gr_chatinterface.chatbot.clear(fn=gr_chatinterface_chatbot_clear_fn, inputs=[gr_dynamic_trigger], outputs=[gr_video_state, gr_state, gr_static_trigger, gr_dynamic_trigger])
            gr_clean_button.click(fn=gr_chatinterface_chatbot_clear_fn, inputs=[gr_dynamic_trigger], outputs=[gr_video_state, gr_state, gr_static_trigger, gr_dynamic_trigger])
            
            # @spaces.GPU
            def gr_for_streaming(history: list[gr.ChatMessage], video_state: dict, state: dict, mode: str, static_trigger: int, dynamic_trigger: int): 
                if static_trigger == 0:
                    yield [], {}, dynamic_trigger
                    return
                global gradio_backend
                if gradio_backend is None:
                    yield history + [gr.ChatMessage(role="assistant", content='(ZeroGPU needs to initialize model under @spaces.GPU, thanks for waiting...)')] , state, dynamic_trigger
                    gradio_backend = GradioBackend(model_path=args.model_path, device=args.device)
                yield history + [gr.ChatMessage(role="assistant", content='(Loading video now... thanks for waiting...)')], state, dynamic_trigger
                if not js_monitor:
                    video_state['video_timestamp'] = 19260817 # 👓
                state.update(video_state)
                query, assistant_waiting_message = None, None
                for message in history[::-1]:
                    if message['role'] == 'user':
                        if message['metadata'] is None or message['metadata'].get('status', '') == '':
                            query = message['content']
                            if message['metadata'] is None:
                                message['metadata'] = {}
                            message['metadata']['status'] = 'pending'
                            continue
                        if query is not None: # put others as done
                            message['metadata']['status'] = 'done'
                    elif message['content'] == '(Loading video now... thanks for waiting...)':
                        assistant_waiting_message = message
                
                for (start_timestamp, stop_timestamp), response, state in gradio_backend(message=query, state=state, mode=mode, hf_spaces=hf_spaces):
                    if start_timestamp >= 0:
                        response_with_timestamp = f'{start_timestamp:.1f}s-{stop_timestamp:.1f}s: {response}'
                        if assistant_waiting_message is None:
                            history.append(gr.ChatMessage(role="assistant", content=response_with_timestamp))
                        else:
                            assistant_waiting_message['content'] = response_with_timestamp
                            assistant_waiting_message = None
                        yield history, state, dynamic_trigger
                if js_monitor:
                    yield history, state, 1 - dynamic_trigger
                else:
                    yield history, state, dynamic_trigger
            
            js_video_timestamp_fetcher = """
                (state, video_state) => {
                    const videoEl = document.querySelector("#gr_video video");
                    return { video_path: videoEl.currentSrc, video_timestamp: videoEl.currentTime };
                }
            """

            def gr_get_video_state(video_state):
                if 'file=' in video_state['video_path']:
                    video_state['video_path'] = video_state['video_path'].split('file=')[1]
                return video_state
            def gr_video_change_fn(mode):
                return [1, 1] if mode == "Real-Time Commentary" else [0, 0]
            gr_video.change(
                fn=gr_video_change_fn, 
                inputs=[gr_radio_mode], 
                outputs=[gr_static_trigger, gr_dynamic_trigger]
            )
            
            gr_dynamic_trigger.change(
                fn=gr_get_video_state,
                inputs=[gr_video_state],
                outputs=[gr_video_state],
                js=js_video_timestamp_fetcher
            ).then(
                fn=gr_for_streaming, 
                inputs=[gr_chatinterface.chatbot, gr_video_state, gr_state, gr_radio_mode, gr_static_trigger, gr_dynamic_trigger], 
                outputs=[gr_chatinterface.chatbot, gr_state, gr_dynamic_trigger], 
            )
    
    demo.queue(max_size=5, default_concurrency_limit=5)
    demo.launch(share=args.share)
