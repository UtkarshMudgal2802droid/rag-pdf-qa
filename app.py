import gradio as gr
from transformers import pipeline

# Load text-generation pipeline
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_length=500,
    truncation=True
)

def generate_story(prompt, story_type="space", max_length=150, temperature=0.85, num_sequences=1):
    """Generate stories with validation."""
    
    # Validation
    if not prompt or len(prompt.strip()) == 0:
        return "Error: Please enter a prompt for the story.", "Error"
    
    if len(prompt.split()) > 50:
        return "Error: Prompt is too long. Please use 50 words or less.", "Error"
    
    if max_length < 20 or max_length > 500:
        return "Error: max_length must be between 20 and 500.", "Error"
    
    if temperature < 0.1 or temperature > 2.0:
        return "Error: temperature must be between 0.1 and 2.0.", "Error"
    
    try:
        enhanced_prompts = {
            "space": f"In the year 2050, robots started {prompt} and explored the vast cosmos...",
            "horror": f"The old house at the end of the street was {prompt} and darkness fell...",
            "motivational": f"Never give up on your dreams because {prompt} and success will follow..."
        }
        
        actual_prompt = enhanced_prompts.get(story_type, prompt)
        
        stories = generator(
            actual_prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_sequences,
            truncation=True,
            pad_token_id=generator.model.config.eos_token_id,
            eos_token_id=generator.model.config.eos_token_id
        )
        
        # Format output with clean text
        story_type_title = story_type.title()
        result = f"""
{story_type_title} Story
{'=' * 60}

{stories[0]['generated_text']}

{'=' * 60}
Generated successfully! ({len(stories)} variation)
"""
        return result, "Success"
    
    except Exception as e:
        return f"Error: {str(e)}", "Error"

def generate_all_stories(max_length=150, temperature=0.85):
    """Generate all 3 required stories."""
    
    output = []
    
    # Space Story
    output.append("SPACE STORY")
    output.append("=" * 60)
    space_prompt = "In the year 2050, robots started"
    try:
        space_story = generator(space_prompt, max_length=max_length, temperature=temperature, num_return_sequences=1, truncation=True, pad_token_id=generator.model.config.eos_token_id)
        output.append(space_story[0]['generated_text'])
    except Exception as e:
        output.append(f"Error: {str(e)}")
    
    output.append("\n" + "=" * 60 + "\n")
    
    # Horror Story
    output.append("HORROR STORY")
    output.append("=" * 60)
    horror_prompt = "The old house at the end of the street was"
    try:
        horror_story = generator(horror_prompt, max_length=max_length, temperature=temperature + 0.05, num_return_sequences=1, truncation=True, pad_token_id=generator.model.config.eos_token_id)
        output.append(horror_story[0]['generated_text'])
    except Exception as e:
        output.append(f"Error: {str(e)}")
    
    output.append("\n" + "=" * 60 + "\n")
    
    # Motivational Paragraph
    output.append("MOTIVATIONAL PARAGRAPH")
    output.append("=" * 60)
    motivational_prompt = "Never give up on your dreams because"
    try:
        motivational = generator(motivational_prompt, max_length=max_length - 20, temperature=temperature - 0.1, num_return_sequences=1, truncation=True, pad_token_id=generator.model.config.eos_token_id)
        output.append(motivational[0]['generated_text'])
    except Exception as e:
        output.append(f"Error: {str(e)}")
    
    output.append("\n" + "=" * 60)
    output.append("All 3 stories generated successfully!")
    
    return "\n".join(output)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown(
        """
        # AI Story Generator
        
        Generate short stories using GPT-2 text-generation pipeline
        
        Create space stories, horror stories, and motivational paragraphs
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Custom Story"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Story Prompt",
                        placeholder="Enter your story prompt (maximum 50 words)...",
                        lines=4,
                        max_lines=8
                    )
                    
                    story_type = gr.Radio(
                        choices=[("Space Story", "space"), ("Horror Story", "horror"), ("Motivational", "motivational")],
                        value="space",
                        label="Story Type"
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        max_length_slider = gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Max Length (tokens)")
                        temperature_slider = gr.Slider(minimum=0.5, maximum=1.2, value=0.85, step=0.05, label="Temperature (Creativity)")
                        num_sequences = gr.Slider(minimum=1, maximum=3, value=1, step=1, label="Number of Variations")
                    
                    generate_btn = gr.Button("Generate Story", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    story_output = gr.Textbox(
                        label="Story Output",
                        lines=18,
                        max_lines=25,
                        show_label=True
                    )
                    
                    story_status = gr.Label(label="Status")
            
            gr.Examples(
                examples=[
                    ["taking over human jobs", "space", 150, 0.85, 1],
                    ["never opened its doors", "horror", 120, 0.9, 1],
                    ["you are capable of greatness", "motivational", 100, 0.75, 1],
                ],
                inputs=[prompt_input, story_type, max_length_slider, temperature_slider, num_sequences],
                outputs=[story_output, story_status],
                fn=generate_story,
            )
        
        with gr.TabItem("Generate All Stories"):
            gr.Markdown("Generate all 3 required stories (Space, Horror, Motivational)")
            
            all_max_length = gr.Slider(minimum=100, maximum=200, value=150, step=10, label="Max Length for All Stories")
            all_temperature = gr.Slider(minimum=0.7, maximum=1.0, value=0.85, step=0.05, label="Temperature for All Stories")
            generate_all_btn = gr.Button("Generate All 3 Stories", variant="primary", size="lg")
            all_stories_output = gr.Textbox(
                label="All Stories",
                lines=22,
                max_lines=30,
                show_label=True
            )
        
        with gr.TabItem("Parameter Guide"):
            gr.Markdown(
                """
                ## Understanding Parameters
                
                | Parameter | Effect | Recommended Range |
                |-----------|--------|-------------------|
                | max_length | Controls story length | 50-300 tokens |
                | temperature | Controls creativity/randomness | 0.5-1.2 |
                | num_return_sequences | Number of story variations | 1-3 |
                
                ### Tips
                
                - Higher temperature (1.0+) = More creative, less predictable
                - Lower temperature (0.5-0.7) = More focused, coherent
                - Longer max_length = More detailed stories
                """
            )
    
    generate_btn.click(fn=generate_story, inputs=[prompt_input, story_type, max_length_slider, temperature_slider, num_sequences], outputs=[story_output, story_status])
    generate_all_btn.click(fn=generate_all_stories, inputs=[all_max_length, all_temperature], outputs=all_stories_output)
    
    gr.Markdown("---\n**GitHub**: https://github.com/UtkarshMudgal/ai-story-generator\n\n**Skills Demonstrated**: max_length, temperature, num_return_sequences")

if __name__ == "__main__":
    demo.launch(share=True)