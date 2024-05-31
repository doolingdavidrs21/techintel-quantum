import pandas as pd
import plotly.express as px
#import altair as alt
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
#from st_aggrid import AgGrid, GridUpdateMode, JsCode
#from st_aggrid.grid_options_builder import GridOptionsBuilder
import networkx as nx
import igraph as ig
from streamlit_plotly_events import plotly_events
import math
import plotly.io as pio
import altair as alt
import pickle
import pydeck as pdk
import os
#from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from itertools import combinations
from pydeck.types import String

# file://wsl.localhost/Ubuntu-22.04/home/davidd/2023/openalex-jamming-gpt4/updatechart.html



os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["MAPBOX_TOKEN"] = st.secrets["MAPBOX_TOKEN"]
MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
#os.environ["OPENAI_API_KEY"] = st.secrets["DB_TOKEN"]

# https://platform.openai.com/docs/models/gpt-3-5

#llm = ChatOpenAI(model_name = 'gpt-4-1106-preview', # 'gpt-3.5-turbo', # 'text-davinci-003' , 'gpt-3.5-turbo'
#             temperature=0.3,
#             max_tokens=600)


llm = ChatOpenAI(model_name = 'gpt-4o', # 'gpt-3.5-turbo', # 'text-davinci-003' , 'gpt-3.5-turbo'
             temperature=0.2,
             max_tokens=3200)

article_template = """
I want you to act as a scientific consultant to help intelligence 
analysts understand the if a given paper poses any kind of concern to 
United States security. 
Act like a Systems Engineering and Technical Assistance (SETA) consultant. 
The summary from you is based on article title, article abstract, the list
of authors, and the list of affiliations. 

Return a brief but detailed description of the scientific topic and applications related to
the scientific field desribed by the title, abstract, list of authors, and the list
of author affiliations. The description should be meaningful to an
new intelligence analyst. Highlight typical applications. Highlight any dual use technologies that may be of concern to the United States
Government. 

What is a good summary of the scientific paper with  title {article_title} and abstract {article_abstract}?
Take into account the list of authors {author_list} and list of affiliations {affiliation_list}. Highlight especially
any collaborations between affiliations in different countries. 
Provide the summary in about 300 words or less. 
Please end with a complete sentence.
"""

prompt_article = PromptTemplate(
    input_variables=["article_title","article_abstract","author_list",
                    "affiliation_list"],
    template=article_template,
)


chain_article = LLMChain(llm=llm, prompt=prompt_article)


def get_article_llm_description(title:str, abstract:str, authors:list, affils:list):
    """
    takes in the key_phrases list, and the article title
    and returns the openai returned description.
    """
    authors = "; ".join(authors)
    affils = "; ".join(affils)
    return chain_article.run(article_title=title,article_abstract=abstract,
                           author_list=authors, affiliation_list=affils )

############################################################################

topic_template = """
I want you to act as a naming consultant for scientific topics based on keyphrases.
Act like a Systems Engineering and Technical Assistance (SETA) consultant. 

Return a brief but detailed description of the scientific topic and applications related to
the scientific field described by the list of keyphrases. The description should be meaningful to an
new intelligence analyst. Highlight typical applications. Highlight any dual use technologies that may be of concern to the United States
Government.

What is a good summary of the scientific topic related to {topic_phrases}?
Provide the summary in about 180 words. 
Please end with a complete sentence.
"""



detailed_topic_template = """
I want you to act as a naming consultant for scientific topics based on article abstracts.
Act like a Systems Engineering and Technical Assistance (SETA) consultant. 

Return a brief but detailed description of the scientific topic and applications related to
the scientific field described by the sample texts. The description should be meaningful to an
new intelligence analyst. Highlight typical applications. Highlight any dual use technologies that may be of concern to the United States
Government.

Provde a bullet list summary of the scientific topic related to these texts: {topic_texts}?
Provide the summary in about 1000 words or less. 
End with a complete sentence; the last character should be a period '.'.
"""




prompt_topic = PromptTemplate(
    input_variables=["topic_phrases"],
  #  input_variables=["topic_texts"],
    template=topic_template,
)


detailed_prompt_topic = PromptTemplate(
   # input_variables=["topic_phrases"],
    input_variables=["topic_texts"],
    template=detailed_topic_template,
)

chain_topic= LLMChain(llm=llm, prompt=prompt_topic)

detailed_chain_topic= LLMChain(llm=llm, prompt=detailed_prompt_topic)

def get_topic_llm_description(key_phrases:list):
    """
    takes in the key_phrases list
    and returns the openai returned description.
    """
  #  st.write(type(key_phrases))
    topic_phrases = ", ".join(key_phrases)
    return chain_topic.run(topic_phrases=topic_phrases)


def get_detailed_topic_llm_description(texts:list):
    """
    takes in the texts list
    and returns the openai returned description.
    """
  #  st.write(type(texts))
    topic_texts = ":: ".join(texts)
    return detailed_chain_topic.run(topic_texts=topic_texts)



#pio.templates.default = "plotly"  
# https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-pandas-dataframe-csv
# https://towardsdatascience.com/how-to-deploy-interactive-pyvis-network-graphs-on-streamlit-6c401d4c99db
pio.templates.default = "plotly_dark"
# https://discuss.streamlit.io/t/streamlit-overrides-colours-of-plotly-chart/34943
st.set_page_config(layout='wide')

st.title("Example: Quantum Technologies flagged by the Defense Science Board")
st.markdown("""
[Applications of Qauntum Technologies, October 2019](https://dsb.cto.mil/reports/2010s/DSB_QuantumTechnologies_Executive%20Summary_10.23.2019_SR.pdf)

* Quantum computer (theoretical computation device relying on quantum mechanics), 量子计算机, Квантовый компьютер
* Quantum cryptography, 量子密碼學, Квантовая криптография
* Quantum sensor, (measurement device using quantum mechanical effects such as entanglement), 
* Topological quantum computer (hypothetical fault-tolerant quantum computer based on topological condensed matter), 拓樸量子電腦
* Adiabatic quantum computation (type of quantum information processing), 绝热量子计算机, Адиабатические квантовые вычисления
* Trapped ion quantum computer (proposed quantum computer implementation), 俘获离子量子计算机
* Post-quantum cryptography (cryptography that is secure against quantum computers), 后量子密码学, постквантовая криптография
* One-way quantum computer (quantum computer that first prepares an entangled resource state and then performs single qubit measurements on it)
""")

st.write("Topic modeling")

@st.cache_data()
def load_centroids_asat():
    #dg = pd.read_csv("penguins.csv", engine="pyarrow")
  #  df = pd.read_json(df.to_json())
    dg = pd.read_pickle('updatejammingcentroids2d.pkl.gz')
    #return dg
    return dg[dg.cluster != -1]

@st.cache_data()
def load_dftriple_asat():
    dg = pd.read_pickle('updatejammingdftriple2d.pkl.gz')
    return dg

@st.cache_data()
def load_dfinfo_asat():
    dg = pd.read_pickle('updatejammingdfinfo2d.pkl.gz')
    #return dg
    return dg[dg['cluster'] != -1]

#@st.cache_data()
#def load_dfgeo_asat():
#    dg = pd.read_pickle('asatgeo.pkl')
#    return dg


@st.cache_data()
def load_source_dict():
    with open("updatesource_page_dict.pkl", "rb") as f:
        source_dict = pickle.load(f)
    return source_dict


@st.cache_data()
def load_affil_geo_dict():
    with open("updateaffil_geo_dict.pkl", "rb") as f:
        affil_geo_dict = pickle.load(f)
    return affil_geo_dict



#@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

#########################################


centroids = load_centroids_asat()
dftriple = load_dftriple_asat()
dfinfo = load_dfinfo_asat()
dfinfo['cluster_'] = dfinfo["cluster"].apply(str)
#dfgeo = load_dfgeo_asat()
source_dict = load_source_dict()
affil_geo_dict = load_affil_geo_dict()

kw_dict = dfinfo['keywords'].to_dict()


color_education = [228,26,28]
color_facility = [55, 126, 184]
color_government = [77, 175, 74]
color_other = [152, 78, 163]
color_nonprofit = [255, 127, 0]
color_company = [255, 255, 51]
color_healthcare = [166, 86, 40]
color_archive = [247, 129, 191]


fill_color_dict = {
    'education': color_education,
    'facility': color_facility,
    'government': color_government,
    'company': color_company,
    'nonprofit': color_nonprofit,
    'other': color_other,
    'healthcare': color_healthcare,
    'archive': color_archive
}

dftriple['fill_color'] = dftriple['type'].map(fill_color_dict)


dftriple['r'] = dftriple['fill_color'].apply(lambda x: x[0])
dftriple['g'] = dftriple['fill_color'].apply(lambda x: x[1])
dftriple['b'] = dftriple['fill_color'].apply(lambda x: x[2])



# add in the affiliations as nodes as well; that row, author, paper, affil. all three get links. ok.
def create_nx_graph(df: pd.DataFrame, cl:int) -> nx.Graph:
    """
    takes the dataframe df, and creates the undirected graph
    from the source and target columns for each row.
    """
    g = nx.Graph() # dc['paper_cluster'] == cl
    dc = df[df['paper_cluster'] == cl]
    author_counts = dc['paper_author_id'].tolist()
    author_counts_dict = {c:author_counts.count(c) for c in author_counts}
    affiliation_counts = dc['id'].tolist()
    affiliation_counts_dict = {c:affiliation_counts.count(c) for c in affiliation_counts}
    source_counts = dc['source'].tolist()
    source_counts_dict = {c:source_counts.count(c) for c in source_counts}
    funder_counts = [x for row in dc['funder_list'].tolist() for x in row]
    funder_counts_dict = {c:funder_counts.count(c) for c in funder_counts}
    for index, row in df[df['paper_cluster'] == cl].iterrows():
        g.add_node(row['paper_id'], group='work', title=row['paper_title'])
        g.add_node(row['paper_author_id'], title=row['paper_author_display_name'],
                   group='author',value = author_counts_dict[row['paper_author_id']])
        try:
            g.add_node(row['id'], group='affiliation',
                   title=row['display_name'] + '\n' + row['country_code'],
                  value = affiliation_counts_dict[row['id']])
        except:
            g.add_node(row['id'], group='affiliation',
                   title=row['display_name'],
                  value = affiliation_counts_dict[row['id']]) 
        if row['source']:
            g.add_node(row['source'], group=row['source_type'],
                      title=row['source'] + ' :\n ' + row['source_type'],
                      value=source_counts_dict[row['source']])
            g.add_edge(
                row['paper_id'],
                row['source'],
            #    title=row['paper_title'] + ' :\n ' + str(row['paper_publication_date']) +  \
            #    ' :\n' + row['source'] + ' :\n ' + \
            #    row['source_type'],
              #  weight = df[(df['paper_id'] == row['paper_id']) & \
              #              (df['source'] == row['source'])]['paper_cluster_score'].sum()
               # weight = row['paper_cluster_score']
            )
            g.add_edge(
                row['paper_author_id'],
                row['source'],
            #    title=row['paper_author_display_name'] + ':\n' + row['source'],
             #   weight = df[(df['paper_author_id'] == row['paper_author_id']) & \
              #              (df['source'] == row['source'])]['paper_cluster_score'].sum()
               # weight = row['paper_cluster_score']
            )
            g.add_edge(
                row['id'],
                row['source'],
             #   title=row['display_name'] + ':\n' + row['source']
            )
        if len(row['funder_list']) > 0:
            for f in row['funder_list']:
                g.add_node(f, group='funder',
                          title=str(f),
                          value=funder_counts_dict[f])
                g.add_edge(
                       row['paper_id'],
                       f,
                  #     title=row['paper_title'] + ':\n ' +  str(row['paper_publication_date']) + \
                  #     ' :\n' + str(f),
                  #  weight = row['paper_cluster_score']
                   )
                g.add_edge(
                       f,
                       row['paper_author_id'],
                    #   title=row['paper_author_display_name'] + ' :\n ' + \
                  #     str(f),
                  #  weight = row['paper_cluster_score']
                       
                   )
                g.add_edge(
                       f,
                       row['id'],
                  #     title=row['display_name'] + '\n' + row['country_code'] + ' :\n ' + \
                   #    str(f)  ,
                  #  weight = row['paper_cluster_score']
                   )  
                if row["source"]:
                    g.add_edge(
                        f,
                        row["source"],
                   #     title=row["source"] + ' :\n' + str(f),
                     #   weight = row['paper_cluster_score']
                    )
        g.nodes[row['paper_id']]['title'] = (
            row['paper_title'] + ' :\n ' + str(row['paper_publication_date'] + ':\n' + 
            '\n'.join(kw_dict[row['paper_id']]))
        )
        g.nodes[row['paper_author_id']]['title'] = (
            row['paper_author_display_name']
        )
        g.add_edge(
            row['paper_id'],
            row['paper_author_id'],
      #  title=row['paper_title'] + ' :\n ' + row['paper_author_display_name'] + ' :\n ' + \
      #      row['paper_raw_affiliation_string'],
         #   weight = row['paper_cluster_score']
        )
        g.add_edge(
            row['paper_author_id'],
            row['id'],
       #     title=row['paper_author_display_name'] + ' :\n ' + \
       #     row['display_name'] + ' :\n ' + row['country_code'],
          #  weight = row['paper_cluster_score']
        )
        g.add_edge(
            row['paper_id'],
            row['id'],
        #    title=row['paper_title'] + ' :\n ' + str(row['paper_publication_date']) + ':\n' + 
        #    row['display_name'] + ' :\n ' + row['country_code'],
         #   weight = row['paper_cluster_score']
        )
        
    g_ig = ig.Graph.from_networkx(g) # assign 'x', and 'y' to g before returning
    #layout = g_ig.layout_auto()
    #layout = g_ig.layout_davidson_harel()
    layout = g_ig.layout_umap(min_dist = 2, epochs = 500)
    # https://igraph.org/python/tutorial/0.9.6/visualisation.html
    coords = layout.coords
    allnodes = list(g.nodes())
    coords_dict = {allnodes[i]:(coords[i][0], coords[i][1]) for i in range(len(allnodes))}
    for i in g.nodes():
        g.nodes[i]['x'] = 250 * coords_dict[i][0] # the scale factor needed 
        g.nodes[i]['y'] = 250 * coords_dict[i][1]
    return g
                
                


#@st.cache_resource()
def create_pyvis_html(cl: int, filename: str = "pyvis_coauthorships_graph.html"):
    """
    wrapper function that calls create_nx_graph to finally 
    produce an interactive pyvis standalone html file
    """
    g_nx = create_nx_graph(dftriple, cl);
    h = Network(height="1000px",
          #  heading="Mitigations and Techniques Relationships",
                width="100%",
                cdn_resources="remote", # can grab the visjs library to make this local if needed
            # probably should
                bgcolor="#222222",
            neighborhood_highlight=True,
              # default_node_size=1,
                font_color="white",
                directed=False,
               # select_menu=True,
                filter_menu=True,
                notebook=False,
               )
    #h.repulsion()
    h.from_nx(g_nx, show_edge_weights=False)
    #h.barnes_hut()
    #h.repulsion(node_distance=40,
    #            central_gravity=-0.2, spring_length=5, spring_strength=0.005, damping=0.09)
    neighbor_map = h.get_adj_list()
   # for node in h.nodes:
   #     if node['group'] == 'author':
   #         a = list(neighbor_map[node["id"]]) # want to insert a "\n" into every third element of a
   #     if node['group'] == 'work':
   #         a = list(neighbor_map[node["id"]])
   #     i = 3
   #     while i < len(a):
   #         a.insert(i, "\n")
   #         i += 4
   #     node["title"] += "\n Neighbors: \n" + " | ".join(a)
   #     node["value"] = len(neighbor_map[node["id"]]) 
# "physics": {
#    "enabled": false
#  },
    h.set_options(
    """
const options = {
  "interaction": {
    "navigationButtons": false
  },
 "physics": {
     "enabled": false
 },
  "edges": {
    "color": {
        "inherit": true
    },
    "setReferenceSize": null,
    "setReference": {
        "angle": 0.7853981633974483
    },
    "smooth": {
        "forceDirection": "none"
    }
  }
  }
    """
    )
    #h.show_buttons(filter_=['physics'])
  #  h.barnes_hut()
    #h.repulsion()
    try:
        path = './tmp'
        h.save_graph(f"{path}/{filename}")
        HtmlFile = open(f"{path}/{filename}","r",
                        encoding='utf-8')
    except:
        h.save_graph(f"{filename}")
        HtmlFile = open(f"{filename}", "r",
                        encoding="utf-8")
    return HtmlFile


#htmlfile = create_pyvis_html()


#st.map(dfgeo)

st.dataframe(centroids[['cluster','x','y','concepts','keywords']])
csv_topics = convert_df(centroids[['cluster','x','y','concepts','keywords']])
st.download_button(
   "Press to Download Topics Table",
   csv_topics,
   "topics.csv",
   "text/csv",
   key='download-topics-csv'
)
#AgGrid(centroids[['cluster','x','y','concepts','keywords']])
# https://medium.com/@hhilalkocak/streamlit-aggrid-6dbbab3afe03
#gd = GridOptionsBuilder.from_dataframe(centroids[['cluster','x','y','concepts','keywords']])
#gd.configure_pagination(enabled=True)
#gridOptions = gd.build()
#AgGrid(centroids[['cluster','x','y','concepts','keywords']],
#       height=500
#      )
#AgGrid(centroids[['cluster','x','y','concepts','keywords']],
#                   fit_columns_on_grid_load=True,
#                   height=500,
#                   width='100%',
#                   theme="streamlit",
#                   reload_data=True,
#                   allow_unsafe_jscode=True
#                  )

#@st.cache_data()
def get_fig_asat():
    fig_centroids = px.scatter(centroids[centroids.cluster != -1],
                           x='x',y='y',
                    color_discrete_sequence=['pink'],
                          hover_data=['x','y',
                                      'wrapped_keywords',
                                      'wrapped_concepts','cluster'])
    fig_centroids.update_traces(marker=dict(size=12,
                              line=dict(width=.5,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    fig_papers = px.scatter(dfinfo[dfinfo.cluster != -1],
                           x='x',y='y',
                    color='cluster_',
                        hover_data = ['title','cluster',
                                      'publication_date'])
                     #     hover_data=['title','x','y',
                     #                 'z','cluster','wrapped_author_list',
                     #                 'wrapped_affil_list',
                     #                 'wrapped_keywords'])
    fig_papers.update_traces(marker=dict(size=4,
                              line=dict(width=.5,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    layout = go.Layout(
        autosize=True,
        width=1000,
        height=1000,

        #xaxis= go.layout.XAxis(linecolor = 'black',
         #                 linewidth = 1,
         #                 mirror = True),

        #yaxis= go.layout.YAxis(linecolor = 'black',
         #                 linewidth = 1,
         #                 mirror = True),

        margin=go.layout.Margin(
            l=10,
            r=10,
            b=10,
            t=10,
            pad = 4
            )
        )

    fig3 = go.Figure(data=fig_papers.data + fig_centroids.data)
    fig3.update_layout(height=700)

                   # layout=layout)  
    return fig3


#centroids = load_centroids()
#dftriple = load_dftriple()
#dfinfo = load_dfinfo()
#dfinfo['cluster_'] = dfinfo["cluster"].apply(str)
bigfig = get_fig_asat()

st.subheader("Papers and Topics", divider='rainbow')
# st.subheader("Feature influence ranking", divider='rainbow')
st.write("Use the navigation tools in the mode bar to pan and zoom. Papers are automatically clustered into subtopics. Topics are the bigger pink dots with representative keywords and phrases available on hover. Clicking on a topic or paper then triggers a report of the most profilic countries, affiliations, and authors associated with that topic.")
selected_point = plotly_events(bigfig, click_event=True, override_height=700)
if len(selected_point) == 0:
    st.write("Select a paper or cluster")
    st.stop()
    
#st.write(selected_point)

selected_x_value = selected_point[0]["x"]
selected_y_value = selected_point[0]["y"]
#selected_species = selected_point[0]["species"]

try:
    df_selected = dfinfo[
        (dfinfo["x"] == selected_x_value)
        & (dfinfo["y"] == selected_y_value)
    ]
    selected_cluster = df_selected['cluster'].iloc[0]
    article_keywords = df_selected['keywords'].to_list()[0]
    article_title = df_selected['title'].iloc[0]
    article_abstract = df_selected['abstract'].iloc[0]
    article_authors = df_selected['author_list'].iloc[0]
    article_affils = df_selected['affil_list'].iloc[0]
    llm_article_description = get_article_llm_description(article_title, article_abstract,
                article_authors,  article_affils)
    st.write(f"Selected Article")
    st.data_editor(
        df_selected[['x', 'y', 'id', 'title', 'doi', 'cluster', 
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','probability']],
        column_config={
            "doi": st.column_config.LinkColumn("doi"),
            "id": st.column_config.LinkColumn("id")
        },
        hide_index=True,
        )
    st.write(llm_article_description)
#st.write(topic_keywords)
except:
    #pass
    selected_cluster_list = centroids[
        (centroids["x"] == selected_x_value)
        & (centroids["y"] == selected_y_value)
    ]['cluster'].to_list()
    if selected_cluster_list:
        selected_cluster = selected_cluster_list[0]


#def make_clickable(url, name):
#    return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(url,name)

#df_selected['link'] = df_selected.apply(lambda x: make_clickable(x['id'], x['id']), axis=1)
#    st.data_editor(
#        df_selected[['x', 'y', 'id', 'title', 'doi', 'cluster', 'probability',
#       'publication_date', 'keywords', 'top_concepts', 'affil_list',
#       'author_list']],
#        column_config={
#            "doi": st.column_config.LinkColumn("doi"),
#            "id": st.column_config.LinkColumn("id")
#        },
#        hide_index=True,
#        )
#    selected_cluster = df_selected['cluster'].iloc[0]
#    st.write(selected_cluster)
#except:
#    df_selected_centroid = centroids[
#        (centroids["x"] == selected_x_value)
#        & (centroids["y"] == selected_y_value)
#    ]
#    selected_cluster = df_selected_centroid['cluster'].iloc[0]
    
    



#st.dataframe(df_selected)
#selected_cluster = df_selected['cluster'].iloc[0]
df_selected_centroid = centroids[
    (centroids['cluster'] == selected_cluster)
]
df_selected_papers = dfinfo[
    (dfinfo['cluster'] == selected_cluster)
].sort_values('probability',ascending=False)
st.write(f"selected topic {selected_cluster}")
st.dataframe(df_selected_centroid[['concepts','keywords','x','y']])

csv_selected_centroid = convert_df(df_selected_centroid[['concepts','keywords',
                                                         'x','y']])
st.download_button(
   "Press to Download Selected Topic",
   csv_selected_centroid,
   "selected_topic.csv",
   "text/csv",
   key='download-selected-topic-csv'
)

# need the information in dfinfo
topic_keywords = df_selected_centroid['keywords'].to_list()[0]
#st.write(topic_keywords)
#llm_topic_description = get_topic_llm_description(topic_keywords)
#st.write(llm_topic_description)

topic_abstracts = dfinfo[dfinfo["cluster"] == selected_cluster]['abstract'].dropna().to_list()[:100]
#st.write(topic_abstracts[:5])
detailed_llm_topic_description = get_detailed_topic_llm_description(topic_abstracts)
st.write(detailed_llm_topic_description)

st.subheader(f"Publications in topic {selected_cluster}", divider='rainbow')
#st.write(f"publications in topic {selected_cluster}")
st.data_editor(
        df_selected_papers[['x', 'y', 'id', 'title', 'doi', 'cluster', 
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','probability']],
        column_config={
            "doi": st.column_config.LinkColumn("doi"),
            "id": st.column_config.LinkColumn("id")
        },
        hide_index=True,
        )

csv_selected_papers = convert_df(df_selected_papers[['x', 'y', 'id', 'title', 'doi', 'cluster', 
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','probability']])

st.download_button(
   f"Press to Download Selected Papers for topic {selected_cluster}",
   csv_selected_papers,
   f"selected_papers_{selected_cluster}.csv",
   "text/csv",
   key='download-selected-topic-papers-csv'
)




def get_country_cluster_sort(dc:pd.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by id, ror sorted
    by the some of probablity descending
    """
    dg = dc[dc['paper_cluster'] == cl].copy()
   # print(cl)
    dv = dg.groupby(['country_code'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    return dv, centroids[centroids.cluster == cl]['keywords'].iloc[0]


def get_affils_cluster_sort(dc:pd.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by id, ror sorted
    by the some of probablity descending
    """
    # https://learning.oreilly.com/library/view/streamlit-for-data/9781803248226/text/ch004.xhtml
    dg = dc[dc['paper_cluster'] == cl].copy()
    print(cl)
    dv = dg.groupby(['id','display_name','country_code',
                     'type','r','g','b'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv.reset_index(inplace=True) # map the display_name column with the geo_dict to get lattitude, longitude
    dv['latitude'] = dv['display_name'].apply(lambda x: affil_geo_dict.get(x, (None, None))[0])
    dv['longitude'] = dv['display_name'].apply(lambda x: affil_geo_dict.get(x, (None, None))[1])
    kw = centroids[centroids.cluster == cl]['keywords'].iloc[0]
    return dv, kw


def get_author_cluster_sort(dc:pd.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by id, ror sorted
    by the some of probablity descending
    """
    dg = dc[dc['paper_cluster'] == cl].copy()
   # print(cl)
    dv = dg.groupby(['paper_author_id','paper_author_display_name',
                    'display_name',
                     'country_code'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv.reset_index(inplace=True)
    return dv, centroids[centroids.cluster == cl]['keywords'].iloc[0]


def get_journals_cluster_sort(dc:pd.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by source (where
    source_type == 'journal') sorted
    by the some of probablity descending
    """
    dg = dc[dc['paper_cluster'] == cl].copy()
    print(cl)
    dv = dg[dg['source_type'] == 'journal'].groupby(['source'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv['journal'] = dv.index
    dv['homepage_url'] = dv['journal'].map(source_dict)
    kw = centroids[centroids.cluster == cl]['keywords'].iloc[0]
    return dv[['journal','homepage_url','paper_cluster_score']], kw


def get_conferences_cluster_sort(dc:pd.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by source (where
    source_type == 'journal') sorted
    by the some of probablity descending
    """
    dg = dc[dc['paper_cluster'] == cl].copy()
    print(cl)
    dv = dg[dg['source_type'] == 'conference'].groupby(['source'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv['conference'] = dv.index
   # dv['homepage_url'] = dv['conference'].map(source_dict)
    kw = centroids[centroids.cluster == cl]['keywords'].iloc[0]
    return dv, kw


def get_country_collaborations_sort(dc:pd.DataFrame, cl:int):
    """
    resticts the dataframe dc to cluster value cl
    and returns the results of paper_id s where there is 
    more than one country_code
    """
    dg = dc[dc['paper_cluster'] == cl].copy()
    dv = dg.groupby('paper_id')['country_code'].apply(lambda x: len(set(x.values))).to_frame()
    dc = dg.groupby('paper_id')['country_code'].apply(lambda x: list(set(x.values))).to_frame()
    dc.columns = ['collab_countries']
    dv.columns = ['country_count']
    dv['collab_countries'] = dc['collab_countries']
    dv.sort_values('country_count',ascending=False, inplace=True)
    di = dfinfo.loc[dv.index].copy()
    di['country_count'] = dv['country_count']
    di['collab_countries'] = dv['collab_countries']
    return di[di['country_count'] > 1]


def get_time_series(dg, cl:int):
    """
    takes dg and the cluster number cl
    and returns a time series chart
    by month, y-axis is the article count
    """
    dftime = dg[dg.cluster == cl][['cluster','probability','publication_date']].copy()
    dftime['date'] = pd.to_datetime(dftime['publication_date'])
    dftime.sort_values('date', inplace=True)
    #by_month = pd.to_datetime(dftime['date']).dt.to_period('M').value_counts().sort_index()
    #by_month.index = pd.PeriodIndex(by_month.index)
    #df_month = by_month.rename_axis('month').reset_index(name='counts')
    return dftime

def generate_subsets(lst):
    return sorted(list(combinations(lst, 2)))

def get_pydeck_chart(dh:pd.DataFrame):
    """
    takes the dataframe dg (dvaffils)
    and returns a pydeck chart
    """
    dg = dh.copy()
    dg = dg.dropna(subset=["longitude","latitude"])
    dg = pd.read_json(dg.to_json())

    mean_lat = dg['latitude'].mean()
    mean_lon = dg['longitude'].mean()
    cl_initial_view = pdk.ViewState(
        latitude = dg['latitude'].iloc[0],
        longitude = dg['longitude'].iloc[0],
        zoom = 11
    )
    sp_layer = pdk.Layer(
        'ScatterplotLayer',
        data = dg,
        get_position = ['longitude','latitude'],
        get_radius = 300
    )
    return cl_initial_view, sp_layer

tab1, tab2, tab3, tab4 , tab5, tab6, tab7, tab8, tab9= st.tabs(["Countries", "Affiliations", "Authors",
                                        "Journals","Conferences",
 "Coauthorship Graph", "Country-Country Collaborations",
                    "time evolution of topic","Affiliation Map"])

dvauthor, kwwuathor = get_author_cluster_sort(dftriple, selected_cluster)
#st.dataframe(dvauthor)

dfcollab = get_country_collaborations_sort(dftriple, selected_cluster)

dvaffils, kwwaffils = get_affils_cluster_sort(dftriple, selected_cluster)
        
dc, kw = get_country_cluster_sort(dftriple, selected_cluster)


dvjournals, kwjournals = get_journals_cluster_sort(dftriple, selected_cluster)

dvconferences, kwconferences = get_conferences_cluster_sort(dftriple, selected_cluster)

htmlfile = create_pyvis_html(selected_cluster)

dftime = get_time_series(dfinfo, selected_cluster)

with tab1:
    st.dataframe(dc)
with tab2:
    st.markdown("highlight and click a value in the **id** column to be given more information")
    st.dataframe(
        dvaffils,
        column_config={
            "id": st.column_config.LinkColumn("id"),
        },
        hide_index=True,
    )
    csv_dvaffils = convert_df(dvaffils)
    st.download_button(
       f"Press to Download Affiliations for topic {selected_cluster}",
       csv_dvaffils,
       f"affils_{selected_cluster}.csv",
       "text/csv",
       key='download-affils-csv'
    )
    #st.dataframe(dvaffils)
with tab3:
    st.write("highlight and click a value in the **paper_author_id** to be given more information")
    st.dataframe(
        dvauthor,
        column_config={
            "paper_author_id": st.column_config.LinkColumn("paper_author_id")
        },
        hide_index=True,
    )
    csv_dvauthor = convert_df(dvauthor)
    st.download_button(
       f"Press to Download Authors for topic {selected_cluster}",
       csv_dvauthor,
       f"authors_{selected_cluster}.csv",
       "text/csv",
       key='download-authors-csv'
    )
    
with tab4:
    st.write("Journals most representative of this cluster")
   # st.dataframe(
   #     dvjournals[['journal','paper_cluster_score']],
   #     hide_index=True
   # )
    st.dataframe(
        dvjournals,
        column_config={
            "homepage_url": st.column_config.LinkColumn("homepage_url")
        },
        hide_index=True,
    )
    csv_dvjournals = convert_df(dvjournals)
    st.download_button(
       f"Press to Download Journals for topic {selected_cluster}",
       csv_dvjournals,
       f"journals_{selected_cluster}.csv",
       "text/csv",
       key='download-journals-csv'
    )

    
with tab5:
    st.write("Conferences most representative of this cluster")
    st.dataframe(
        dvconferences[['conference','paper_cluster_score']],
        hide_index=True
    )
    csv_dvconferences = convert_df(dvconferences)
    st.download_button(
       f"Press to Download Conferences for topic {selected_cluster}",
       csv_dvauthor,
       f"conferences_{selected_cluster}.csv",
       "text/csv",
       key='download-conferences-csv'
    )
  #  st.data_editor(
  #      dvconferences,
  #      column_config={
  #          "homepage_url": st.column_config.LinkColumn("homepage_url")
  #      },
  #      hide_index=True,
  #  )
    
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9266366

with tab6:
    st.write("Coauthorship Graph (Papers and Authors)")
    components.html(htmlfile.read(), height=1100)
    
with tab7:
    st.write("Country-Country Collaborations")
    st.dataframe(
        dfcollab[['x', 'y', 'id','collab_countries', 'title', 'doi', 'cluster', 'probability',
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','funder_list']],
        column_config={
            "doi": st.column_config.LinkColumn("doi"),
        },
        hide_index=True,
    )
    csv_dvcollab = convert_df(dfcollab[['x', 'y', 'id','collab_countries', 'title', 'doi', 'cluster', 'probability',
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','funder_list']])
    st.download_button(
       f"Press to Download Country-Country Collab for topic {selected_cluster}",
       csv_dvcollab,
       f"collab_{selected_cluster}.csv",
       "text/csv",
       key='download-collab-csv'
    )
    
with tab8:
    alt_chart= alt.Chart(dftime).mark_line().transform_fold(
    ['probability']
        ).encode(
        x = 'yearmonth(date):T',
        y = 'sum(value):Q',
        color='key:N'
    ).interactive()
    st.altair_chart(alt_chart, use_container_width=True)
    
    
with tab9:
    dg = dvaffils.copy()
    dg = dg.dropna(subset=["longitude","latitude"])
    dg['size'] = 100*dg['paper_cluster_score']
    dg = pd.read_json(dg.to_json())

    mean_lat = dg['latitude'].mean()
    st.write(dg.head())
    mean_lon = dg['longitude'].mean()
    cl_initial_view = pdk.ViewState(
        latitude = dg['latitude'].mean(),
        longitude = dg['longitude'].mean(),
        zoom = 3
    )
    view = pdk.data_utils.compute_view(dg[["longitude", "latitude"]])
    view.pitch = 75
    view.bearing = 60
    da = dftriple[dftriple['paper_cluster'] == selected_cluster].copy()
    dv = da.groupby('paper_id')['display_name'].apply(lambda x: len(set(x.values))).to_frame()
    dc = da.groupby('paper_id')['display_name'].apply(lambda x: list(set(x.values))).to_frame()
    dc.columns = ['collab_affils']
    dv.columns = ['affil_count']
    dv['collab_affils'] = dc['collab_affils']
    dv.sort_values('affil_count', ascending=False, inplace=True)
    dv = dv[dv['affil_count'] > 1].copy()
    dv['subsets'] = dv['collab_affils'].apply(generate_subsets)
    flattened_df = dv.explode('subsets').copy()
    flattened_df['source_affil'] = flattened_df['subsets'].apply(lambda x: x[0])
    flattened_df['target_affil'] = flattened_df['subsets'].apply(lambda x: x[1])
    dfarc = flattened_df['subsets'].value_counts(dropna=False).to_frame().copy()
    dfarc.rename(columns={'subsets': 'count'}, inplace=True)
    dfarc['affils'] = dfarc.index
    dfarc['source'] = dfarc['affils'].apply(lambda x: x[0])
    dfarc['target'] = dfarc['affils'].apply(lambda x: x[1])
    dfarc['source_geo'] = dfarc['source'].map(affil_geo_dict)
    dfarc['target_geo'] = dfarc['target'].map(affil_geo_dict)
    pattern = r"(-?\d+\.\d+), (-?\d+\.\d+)"
    dfarc[['source_lat', 'source_lon']] = dfarc['source_geo'].apply(str).str.extract(pattern)
    dfarc[['target_lat', 'target_lon']] = dfarc['target_geo'].apply(str).str.extract(pattern)
    dfarc['source_lon'] = dfarc['source_lon'].apply(float)
    dfarc['source_lat'] = dfarc['source_lat'].apply(float)
    dfarc['target_lon'] = dfarc['target_lon'].apply(float)
    dfarc['target_lat'] = dfarc['target_lat'].apply(float)
    GREEN_RGB = [0, 255, 0]
    RED_RGB = [240, 100, 0]

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=dfarc.dropna(),
        get_width = "count * 2",
        get_source_position = ['source_lon', 'source_lat'],
        get_target_position = ['target_lon','target_lat'],
        get_tilt=0,
        pickable=True,
        get_source_color=RED_RGB,
        get_target_color=GREEN_RGB,
        auto_highlight = True
    )
    
    sp_layer = pdk.Layer(
        'ScatterplotLayer',
        data = dg,
        get_position = ['longitude','latitude'],
        radius_scale = 75,
        radius_min_pixels=5,
        radius_max_pixels=300,
        line_width_min_pixels=1,
       # get_radius = 300,
        get_radius = "size",
        pickable=True,
        opacity = 0.4,
      #  get_fill_color = ['paper_cluster_score <= 1 ? 255 ? 
        get_fill_color = [65, 182, 196]
    )
    affil_layer = pdk.Layer(
        "ColumnLayer",
        data = dg,
        get_position=["longitude","latitude"],
        get_elevation="size",
        elevation_scale = 200,
       # radius_scale = 75,
       # radius_min_pixels=5,
       # radius_max_pixels=300,
        radius = 3_000,
        line_width_min_pixels=1,
        get_radius="size",
   # radius = 20,
       # get_fill_color=[180, 0, 200, 140],
        get_fill_color=['r','g','b'],
        auto_highlight=True,
        pickable=True,
    )
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=dg,
        opacity=0.8,
        get_position=['longitude','latitude'],
        aggregation=String('MAX'),
        get_weight='paper_cluster_score'
    )
    
    st.pydeck_chart(pdk.Deck(
        layers = [sp_layer, affil_layer, heatmap_layer, arc_layer],
        api_keys = {'mapbox': MAPBOX_TOKEN},
        map_provider='mapbox',
       # map_style="mapbox:styles/mapbox/satellite-streets-v11",
        map_style="mapbox://styles/mapbox/dark-v10",
       #  map_style="mapbox://styles/mapbox/satellite-streets-v11",
       # map_style='dark',
        #tooltip=True,
        initial_view_state=view,
        tooltip = {
            "html": "<b>{display_name}</b> <br/> <b>Strength</b>: {paper_cluster_score} <br>" + \
            "<b>source: {source} <br/> <b>target: {target} <br>" + \
            "<b>count: {count} <br/>",
            "style": {
                "backgroundColor": "white",
                "color": "black"
            }
        }

        #map_style='dark',
        #initial_view_state = cl_initial_view,
        #layers = [sp_layer],
      #  tooltip = {
      #  "html": "<b>{display_name}</b> <br/> <b>Score</b>: {paper_cluster_score} <br>" + \
      #      "<b>{source}</b> <br/> <b>{target}</b> <br/> <b>{count}</b>"
      #  }
        #tooltip = {
        #    "html": "<b>{display_name}</b> <br/> <b>Strength</b>: {paper_cluster_score} <br>" + \
        #    "<b>source: {source} <br/> <b>target</b> {target}"
        #}
    ))