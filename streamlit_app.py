import os
import json
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import search_tool, wiki_tool, save_tool

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Pydantic model
class Response(BaseModel):
    country: str
    cities: list[str]
    duration_days: int
    budget_inr: float
    itinerary: list[str]
    activities: list[str]

parser = PydanticOutputParser(pydantic_object=Response)

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0, 
    google_api_key=api_key
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a vacation planner agent. Use tools if needed, then respond in the following JSON format:

{{
  "country": string,
  "cities": [string],
  "duration_days": int,
  "budget_inr": float,
  "itinerary": [string],
  "activities": [string]
}}

Only output JSON. No explanations.
"""),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm_gemini,
    prompt=prompt,
    tools=tools,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.set_page_config(page_title="Vacation Planner", layout="wide")
st.title("AI Vacation Planner")
st.write("Plan your perfect vacation with AI assistance!")

query = st.text_area("Describe your dream vacation:", placeholder="Example: Plan a 7-day cultural trip to Japan in spring with a budget of ₹2,50,000...")

if st.button("Plan My Trip"):
    if not query.strip():
        st.warning("Please enter your vacation request.")
    else:
        with st.spinner("Planning your trip..."):
            raw_response = agent_executor.invoke({"query": query})
            raw_output = raw_response.get("output", "")

            parsed_response = parser.parse(raw_output)
            st.subheader("Vacation Plan")
            st.markdown(f"**Country:** {parsed_response.country}")
            st.markdown(f"**Cities:** {', '.join(parsed_response.cities)}")
            st.markdown(f"**Duration:** {parsed_response.duration_days} days")
            st.markdown(f"**Budget:** ₹{parsed_response.budget_inr:,.2f}")

            st.subheader("Itinerary")
            for day in parsed_response.itinerary:
                st.write(f"- {day}")

            st.subheader("Activities")
            st.write(", ".join(parsed_response.activities))

            with st.expander("Raw JSON Output"):
                st.write(raw_output)
            
            # Download plan txt 
            trip_text = f"""
VACATION PLAN
=============
Country: {parsed_response.country}
Cities: {', '.join(parsed_response.cities)}
Duration: {parsed_response.duration_days} days
Budget: ₹{parsed_response.budget_inr:,.2f}

Itinerary:
----------
""" + "\n".join([f"{day}" for _,day in enumerate(parsed_response.itinerary)]) + """

Activities:
-----------
""" + "\n".join([f"- {activity}" for activity in parsed_response.activities])

            st.download_button(
                label="Download",
                data=trip_text,
                file_name=f"trip_plan_{parsed_response.country.lower()}.txt",
            )

