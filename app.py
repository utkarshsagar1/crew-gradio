import gradio as gr
import os
from researcher import create_researcher, create_research_task, run_research
from dotenv import load_dotenv

# Handle SQLite for ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

# Load environment variables from .env file
load_dotenv()

def research_process(provider, openai_api_key, groq_api_key, exa_api_key, 
                    openai_model, groq_model, ollama_model, task_description):
    """
    Execute the research process with the given parameters
    
    Args:
        provider (str): The LLM provider (OpenAI, GROQ, or Ollama)
        openai_api_key (str): OpenAI API key (if provider is OpenAI)
        groq_api_key (str): GROQ API key (if provider is GROQ)
        exa_api_key (str): EXA API key for search tools
        openai_model (str): Selected OpenAI model
        groq_model (str): Selected GROQ model
        ollama_model (str): Selected Ollama model
        task_description (str): The research task to perform
        
    Returns:
        tuple: (status message, research results)
    """
    # Set environment variables
    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = openai_api_key
        model = openai_model
    elif provider == "GROQ":
        os.environ["GROQ_API_KEY"] = groq_api_key
        model = groq_model
    elif provider == "Ollama":
        model = ollama_model
    
    # Always set EXA API key if provided
    if exa_api_key:
        os.environ["EXA_API_KEY"] = exa_api_key
    
    # Check required keys
    if provider == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
        return "⚠️ OpenAI API key is required", ""
    elif provider == "GROQ" and not os.environ.get("GROQ_API_KEY"):
        return "⚠️ GROQ API key is required", ""
    
    # For non-Ollama providers, check EXA key
    if provider != "Ollama" and not os.environ.get("EXA_API_KEY"):
        return "⚠️ EXA API key is required for using search tools", ""
    
    # For Ollama, make sure model is selected
    if provider == "Ollama" and not model:
        return "⚠️ No Ollama models found. Please make sure Ollama is running", ""
    
    # Prepare selection dict to match the original code
    selection = {
        "provider": provider,
        "model": model
    }
    
    try:
        # Create researcher agent
        researcher = create_researcher(selection)
        
        # Create research task
        task = create_research_task(researcher, task_description)
        
        # Run the research
        result = run_research(researcher, task)
        
        # Return success message and result
        return "✅ Research completed!", str(result)
    except Exception as e:
        return f"❌ Error occurred: {str(e)}", ""

def create_interface():
    """Create the Gradio interface for the CrewAI Research Assistant"""
    
    with gr.Blocks(title="CrewAI Research Assistant", css="footer {display: none !important}") as app:
        # Create header with logo
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image("https://cdn.prod.website-files.com/66cf2bfc3ed15b02da0ca770/66d07240057721394308addd_Logo%20(1).svg", 
                         show_label=False, height=80)
        
        # Main title
        gr.Markdown("# 🔍 <span style='color: #FF4B4B;'>CrewAI</span> Research Assistant")
        
        with gr.Row():
            # Left side (configuration)
            with gr.Column(scale=1):
                gr.Markdown("# Configuration")
                
                # Model Selection accordion
                with gr.Accordion("🤖 Model Selection", open=True):
                    gr.Markdown("Select LLM Provider:")
                    
                    provider = gr.Radio(
                        choices=["OpenAI", "GROQ", "Ollama"],
                        value="OpenAI",
                        label="Provider"
                    )
                    
                    # OpenAI models
                    with gr.Group(visible=True) as openai_group:
                        gr.Markdown("Select OpenAI Model")
                        openai_model = gr.Dropdown(
                            choices=["GPT-3.5", "GPT-4", "o1", "o1-mini", "o1-preview"],
                            value="GPT-3.5",
                            label=""
                        )
                    
                    # GROQ models
                    with gr.Group(visible=False) as groq_group:
                        gr.Markdown("Select GROQ Model")
                        groq_model = gr.Dropdown(
                            choices=["llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"],
                            value="llama2-70b-4096",
                            label=""
                        )
                    
                    # Ollama models
                    with gr.Group(visible=False) as ollama_group:
                        gr.Markdown("Select Ollama Model")
                        ollama_model = gr.Dropdown(
                            choices=["llama2", "mistral", "gemma", "phi"],
                            value="llama2",
                            label=""
                        )
                
                # API Keys accordion
                with gr.Accordion("🔑 API Keys", open=True):
                    gr.Markdown("API keys are stored temporarily in memory and cleared when you close the browser.")
                    
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        value=os.environ.get("OPENAI_API_KEY", "")
                    )
                    
                    groq_api_key = gr.Textbox(
                        label="GROQ API Key",
                        placeholder="gsk_...",
                        type="password",
                        visible=False,
                        value=os.environ.get("GROQ_API_KEY", "")
                    )
                    
                    exa_api_key = gr.Textbox(
                        label="EXA API Key",
                        placeholder="exa-...",
                        type="password",
                        value=os.environ.get("EXA_API_KEY", "")
                    )
                
                # About accordion
                with gr.Accordion("ℹ️ About", open=False):
                    gr.Markdown("""
                    # CrewAI Research Assistant
                    
                    This app uses CrewAI to perform research on any topic you provide.
                    
                    The research agent uses:
                    1. Exa for web search and information gathering
                    2. Your selected LLM for analysis and report generation
                    
                    Enter any research topic and get a comprehensive report with findings, analysis, and recommendations.
                    """)
            
            # Right side (research interface)
            with gr.Column(scale=2):
                # Research input
                task_description = gr.Textbox(
                    label="What would you like to research?",
                    placeholder="Enter your research topic here...",
                    value="Research the latest AI Agent news in February 2025 and summarize each.",
                    lines=3
                )
                
                # Start button row
                with gr.Row():
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=1):
                        start_button = gr.Button("🚀 Start Research", variant="primary")
                    with gr.Column(scale=1):
                        pass
                
                # Status message
                status_message = gr.Markdown("")
                
                # Research output with download button
                research_output = gr.Markdown(label="Research Results")
                
                # Download section
                download_file = gr.File(
                    label="Download Research Report", 
                    file_count="single", 
                    visible=False
                )
        
        # Footer
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                gr.Markdown("Made with ❤️ using [CrewAI](https://crewai.com), [Exa](https://exa.ai) and [Gradio](https://gradio.app)")
            with gr.Column(scale=1):
                pass
        
        # Show/hide appropriate model selection and API key fields based on provider selection
        def update_visibility(api_choice):
            # Model visibility updates
            openai_visible = gr.update(visible=(api_choice == "OpenAI"))
            groq_visible = gr.update(visible=(api_choice == "GROQ"))
            ollama_visible = gr.update(visible=(api_choice == "Ollama"))
            
            # API key visibility updates
            openai_key_visible = gr.update(visible=(api_choice == "OpenAI"))
            groq_key_visible = gr.update(visible=(api_choice == "GROQ"))
            
            return [openai_visible, groq_visible, ollama_visible, openai_key_visible, groq_key_visible]
        
        provider.change(
            fn=update_visibility,
            inputs=[provider],
            outputs=[openai_group, groq_group, ollama_group, openai_api_key, groq_api_key]
        )
        
        # Function to create a downloadable file
        def create_markdown_file(content):
            if not content:
                return None
                
            file_path = "research_report.md"
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return file_path
            except Exception as e:
                print(f"Error saving file: {e}")
                return None
        
        # Set up research button click handler
        start_button.click(
            fn=research_process,
            inputs=[
                provider, openai_api_key, groq_api_key, exa_api_key,
                openai_model, groq_model, ollama_model, task_description
            ],
            outputs=[status_message, research_output],
            show_progress=True
        ).then(
            fn=create_markdown_file,
            inputs=[research_output],
            outputs=[download_file]
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[download_file]
        )
    
    return app

# Launch the Gradio app
if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=8050, share=True)
