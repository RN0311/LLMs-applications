import seaborn as sns
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def analyze_and_visualize(openai_api_key):
    flights = sns.load_dataset('flights')
    
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    prompt = PromptTemplate(
        input_variables=["data_description"],
        template="""Given this flights dataset info:
        {data_description}
        Suggest 3 visualizations that would best reveal patterns in the data.
        Format: JSON array with [{{type: 'plot_type', x: 'column', y: 'column', title: 'title'}}]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    data_info = f"Columns: {flights.columns.tolist()}\nSample:\n{flights.head()}"
    suggestions = chain.invoke({"data_description": data_info})
    print(f"AI Suggestions: {suggestions['text']}")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.lineplot(data=flights, x='year', y='passengers', ax=axes[0])
    axes[0].set_title('Passenger Trends Over Time')
    
    sns.boxplot(data=flights, x='month', y='passengers', ax=axes[1])
    axes[1].set_title('Monthly Passenger Distribution')
    
    monthly_avg = flights.groupby('month')['passengers'].mean()
    sns.barplot(x=monthly_avg.index, y=monthly_avg.values, ax=axes[2])
    axes[2].set_title('Average Monthly Passengers')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_and_visualize(OPENAI_API_KEY)