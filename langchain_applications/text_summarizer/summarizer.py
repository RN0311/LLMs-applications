from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

OPENAI_API_KEY="" #Add your own API key here


def create_summaries():
  llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
  prompt=PromptTemplate.from_template("""
  Summarize the following text in 3 key points:

  {text}

  KEY POINTS:
  """)

  chain = prompt | llm

  return chain

if __name__=="__main__":
  summarizer = create_summaries()

  text="""
  NASA awarded new study contracts Thursday to help support life and work on the lunar surface. As part of the agency’s blueprint for deep space exploration to support the Artemis campaign, nine American companies in seven states are receiving awards.
  The Next Space Technologies for Exploration Partnerships Appendix R contracts will advance learning in managing everyday challenges in the lunar environment identified in the agency’s Moon to Mars architecture. 
  “These contract awards are the catalyst for developing critical capabilities for the Artemis missions and the everyday needs of astronauts for long-term exploration on the lunar surface,” said Nujoud Merancy, deputy associate administrator, Strategy and Architecture Office at NASA Headquarters in Washington. “The strong response to our request for proposals is a testament to the interest in human exploration and the growing deep-space economy. This is an important step to a sustainable return to the Moon that, along with our commercial partners, will lead to innovation and expand our knowledge for future lunar missions, looking toward Mars.”
  
  The selected proposals have a combined value of $24 million, spread across multiple companies, and propose innovative strategies and concepts for logistics and mobility solutions including advanced robotics and autonomous capabilities:

  Blue Origin, Merritt Island, Florida – logistical carriers; logistics handling and offloading; logistics transfer; staging, storage, and tracking; surface cargo and mobility; and integrated strategies
  Intuitive Machines, Houston, Texas – logistics handling and offloading; and surface cargo and mobility
  Leidos, Reston, Virginia – logistical carriers; logistics transfer; staging, storage, and tracking; trash management; and integrated strategies
  Lockheed Martin, Littleton, Colorado – logistical carriers; logistics transfer; and surface cargo and mobility
  MDA Space, Houston – surface cargo and mobility
  Moonprint, Dover, Delaware – logistical carriers
  Pratt Miller Defense, New Hudson, Michigan – surface cargo and mobility
  Sierra Space, Louisville, Colorado – logistical carriers; logistics transfer; staging, storage, and tracking; trash management; and integrated strategies
  Special Aerospace Services, Huntsville, Alabama – logistical carriers; logistics handling and offloading; logistics transfer; staging, storage, and tracking; trash management; surface cargo and mobility; and integrated strategies
  
  NASA is working with industry, academia, and the international community to continuously evolve the blueprint for crewed exploration and taking a methodical approach to investigating solutions that set humanity on a path to the Moon, Mars, and beyond.
  """
  summary = summarizer.invoke({"text": text})
  print(summary.content)

