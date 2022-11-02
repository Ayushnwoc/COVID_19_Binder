import matplotlib.pyplot as plt
import networkx as nx

# smile = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"

def plot(G, labels, file_):
    plt.cla()
    pos = nx.spring_layout(G, seed = 3128387843)  # positions for all nodes
    
    options = {
        "node_size": 300,
        "node_color": "white"
    }
    nx.draw_networkx_nodes(G, pos, **options)

    options = {
        "edge_color": "black",
        "width": 1
    }
    nx.draw_networkx_edges(G, pos, **options)

    options = {
        "font_size": 10,
        "font_color": "black"
    }
    nx.draw_networkx_labels(G, pos, labels, **options)

    plt.tight_layout()
    plt.axis("off")
    plt.savefig(file_)