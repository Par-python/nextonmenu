"""Gradio demo: type an ingredient, get its lifecycle-stage diagnosis."""
import gradio as gr
from src.diagnose import diagnose


def run(ingredient):
    if not ingredient or not ingredient.strip():
        return "Type an ingredient (e.g. pandan).", "", None
    try:
        headline, rationale, prob, chart = diagnose(ingredient.strip())
        return headline, f"{rationale}\n\n(early-curve match: {prob:.0%})", chart
    except Exception as e:
        return "Could not diagnose this ingredient.", str(e), None


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Ingredient", placeholder="pandan"),
    outputs=[
        gr.Textbox(label="Diagnosis"),
        gr.Textbox(label="Why"),
        gr.Plot(label="Google Trends"),
    ],
    title="NextOnMenu — Is it the next matcha?",
    description="Diagnoses where a food ingredient sits in its trend lifecycle. "
                "Radar, not crystal ball.",
)

if __name__ == "__main__":
    demo.launch()
