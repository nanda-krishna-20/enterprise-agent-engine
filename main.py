import os
from ddtrace import patch_all
from ddtrace.llmobs import LLMObs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Patch standard libraries (HTTP requests, generic database calls) for automatic tracing and observability in Datadog.
patch_all() 

# Enable Datadog Observability for LLMs (only if DD_API_KEY is provided)
dd_api_key = os.environ.get("DD_API_KEY")
if dd_api_key:
    LLMObs.enable(
        ml_app=os.environ.get("DD_LLMOBS_ML_APP", "enterprise-agent-engine"),
        api_key=dd_api_key,
        site=os.environ.get("DD_SITE", "datadoghq.com")
    )

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from crewai import Agent, Task, Crew, Process, LLM
from src.tools import search_real_estate_db


# API Keys should be loaded from environment variables
# Set GROQ_API_KEY, OPENAI_API_KEY, HF_TOKEN in .env or system environment
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.1
)

# 1. Define the State (The Shared Notebook)
class AgentState(TypedDict):
    topic: str
    research_notes: str
    is_approved: bool
    final_report: str

# 2. Define the Nodes (The Workstations)
def researcher_node(state: AgentState):
    print(f"--- [Node] Researcher Team is investigating: {state['topic']} ---")

    # 1. Define the specialist Agent for research
    # llm = os.getenv("LLM_MODEL", "gpt-4")  #
    researcher = Agent(
        role='Senior Research Analyst',
        goal=f'Provide a detailed technical summary about {state['topic']}',
        constraints=[
            'Use only credible sources.',
            'Summarize findings in a clear and concise manner.',
            'Focus on technical aspects and implications.'
        ],
        backstory="You are an expert at breaking down complex AI concepts into simple summaries.",
        performance_evaluation=[
            'The summary should be comprehensive and accurate.',
            'The language should be clear and accessible to non-experts.',
            'The summary should highlight key technical insights and implications.'
        ],
        tools=[search_real_estate_db],
        allow_delegation=False,
        llm=llm
    )

    # 2. Define the Manager (the Hierarchical Pattern)
    manager = Agent(
        role='Research Manager',
        goal='Oversee the research process and ensure the final draft is high quality.',
        constraints=[
            'Ensure the research is thorough and well-organized.',
            'Provide feedback to the Researcher to improve the draft.',
            'Make sure the final draft is ready for human review.'
        ],
        backstory="You coordinate specialists and verify that their work is accurate and well-presented.",
        allow_delegation=True,
        llm=llm
    )

    # 3. Define the Task for the Researcher
    research_task = Task(
        description=f"Conduct deep research on {state['topic']}. Focus on architecture and use cases",
        expected_output="A 2-paragraph technical research summary.",
        agent=researcher
    )

    # 4. Assemble the Crew with the hierarchical process
    crew = Crew(
        agents=[researcher, manager],
        tasks=[research_task],
        process=Process.hierarchical,
        manager_llm=llm
    )

    # 5. Kickoff the Crew and get the result
    result = crew.kickoff()

    # Return the result back to the Shared Notebook (the State)
    return {"research_notes": str(result)}

    # For now, we mock the CrewAI output. In Iteration 2, we will call CrewAI here.
    # return {"research_notes": f"Deep research on {state['topic']} is complete."}

def human_approval_node(state: AgentState):
    print("--- [Node] Waiting for Human Approval ---")
    # This node is a placeholder for the "Pause" point.

    answer = interrupt("Do you approve the research draft? (yes/no)")


    print(f"--- [Node] Human provided answer: {answer} ---")
    return {"is_approved": answer.strip().lower() == "yes"}
   

def writer_node(state: AgentState):
    print("--- [Node] Writer is drafting report ---")
    report = f"FINAL REPORT: {state['research_notes']}"
    return {"final_report": report}

def route_after_human(state: AgentState):
    print(f"--- [Router] Routing based on approval: {state['is_approved']} ---")
    if state["is_approved"]:
        return "writer" # Go to the writer
    else:
        print("--- [Router] Draft rejected. Sending back to Research Team! ---")
        return "researcher" # Loop back to the beginning

# 3. Build the Assembly Line (The Graph)
builder = StateGraph(AgentState)

# Add Workstations
builder.add_node("researcher", researcher_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("writer", writer_node)

# Connect with Conveyor Belts (Edges)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "human_approval")
builder.add_conditional_edges("human_approval", route_after_human)
builder.add_edge("writer", END)

# 4. Compile with Persistence (The Memory Saver)
# This allows the graph to "save its game" at the checkpoint
memory = MemorySaver()
app = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "demo-1"}}

def run_workflow():
    print("\n--- Starting Engine ---")
    
    # 1. Initial Kickoff (We don't need to save the result variable anymore)
    app.invoke(
        {
            "topic": "What is the current valuation and cap rate for industrial logistics hubs near the DFW airport?",
            "research_notes": "",
            "is_approved": False,
            "final_report": ""
        }, 
        config
    )
    
    # 2. The Robust Event Loop
    while True:
        # Ask LangGraph for the exact status of the engine
        current_state = app.get_state(config)
        
        # If 'next' is empty, there are no more nodes to run. The graph is done!
        if not current_state.next:
            break
            
        # Check if the graph is specifically paused at an interrupt
        if current_state.tasks and current_state.tasks[0].interrupts:
            interrupt_msg = current_state.tasks[0].interrupts[0].value
            print(f"\nPAUSED: {interrupt_msg}")
            
            user_input = input("Your Response: ")
            
            print("\n--- Resuming Engine ---")
            # Resume execution with the user's input
            app.invoke(Command(resume=user_input), config)
        else:
            # Failsafe: if it pauses for an unknown reason, break to avoid infinite loop
            break

    # 3. Print the Final Output ONLY when the while loop breaks
    final_data = app.get_state(config).values
    print("\n--- Final Output ---")
    if final_data.get("final_report"):
        print(final_data["final_report"])
    else:
        print("No report was generated (Engine terminated early).")
        

if __name__ == "__main__":
    run_workflow()

