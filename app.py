import gradio as gr
import inference_2 as inference


title="Multimodal deepfake detector"
description="Deepfake detection for video and audio modalities."
            
           
video_interface = gr.Interface(inference.deepfakes_video_predict,
                    gr.Video(),
                    "text",
                    examples = ["videos/celeb_synthesis.mp4", "videos/real-1.mp4"],
                    cache_examples = False
                    )

audio_interface = gr.Interface(inference.deepfakes_spec_predict,
                               gr.Audio(),
                               "text",
                               examples = ["audios/DF_E_2000027.flac", "audios/DF_E_2000031.flac"],
                               cache_examples = False)


app = gr.TabbedInterface(interface_list= [video_interface, audio_interface],
                         tab_names = ['Video Detection', 'Audio Detection'])

if __name__ == '__main__':
    app.launch(share = False)
