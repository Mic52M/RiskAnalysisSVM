import matplotlib.pyplot as plt
import networkx as nx

def create_risk_assignment_flowchart(output_path="risk_assignment_flowchart.png"):
    # Creazione del grafo
    G = nx.DiGraph()
    
    # Definizione dei nodi con etichette
    nodes = {
        "Start": "Dati in Input",
        "Step1": "Addestramento SVM iniziale",
        "Step2": "Identificazione dei Support Vectors",
        "Step3": "Assegnazione Livello di Rischio",
        "Step4": "Rimozione Support Vectors",
        "Step5": "Riaddestramento SVM",
        "Decision": "Tutti i dati classificati?",
        "End": "Dati assegnati ai livelli di rischio"
    }
    
    # Aggiunta dei nodi al grafo
    for key, label in nodes.items():
        G.add_node(key, label=label)
    
    # Definizione degli archi (connessioni tra i nodi)
    edges = [
        ("Start", "Step1"),
        ("Step1", "Step2"),
        ("Step2", "Step3"),
        ("Step3", "Step4"),
        ("Step4", "Step5"),
        ("Step5", "Decision"),
        ("Decision", "End"),
        ("Decision", "Step2")  # Loop finché non sono assegnati tutti i dati
    ]
    
    # Aggiunta degli archi al grafo
    G.add_edges_from(edges)
    
    # Posizionamento dei nodi
    pos = {
        "Start": (0, 5),
        "Step1": (0, 4),
        "Step2": (0, 3),
        "Step3": (0, 2),
        "Step4": (0, 1),
        "Step5": (0, 0),
        "Decision": (2, 1.5),
        "End": (4, 1.5)
    }
    
    # Definizione dei colori per i tipi di nodi
    node_colors = {
        "Start": "lightblue",   # Input
        "Step1": "lightgreen",  # Processo
        "Step2": "lightgreen",
        "Step3": "lightgreen",
        "Step4": "lightgreen",
        "Step5": "lightgreen",
        "Decision": "orange",   # Decisione
        "End": "lightcoral"     # Output
    }
    
    # Disegno del grafo
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=False, node_size=3500, edge_color="gray", font_size=10, font_weight="bold")
    
    # Aggiunta delle etichette personalizzate
    labels = {node: data["label"] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold")
    
    # Colorazione dei nodi
    node_colors_list = [node_colors[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=3500, edgecolors='black')
    
    # Titolo
    plt.title("Diagramma di Flusso: Assegnazione dei Livelli di Rischio", fontsize=12, fontweight="bold")
    
    # Salvataggio dell'immagine
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.show()

# Esegui la funzione per generare il diagramma
create_risk_assignment_flowchart()
