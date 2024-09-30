import torch
from transformers import AutoModel, AutoTokenizer
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP

DEVICE = "mps"
model_name = "sentence-transformers/all-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to(DEVICE)

def get_embeddings_across_layers(sentences, layer_indices, task_prefix="clustering"):
    prefixed_sentences = [f"{task_prefix}: {sentence}" for sentence in sentences]
    inputs = tokenizer(
        prefixed_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    if not getattr(model.config, 'output_hidden_states', False):
        model.config.output_hidden_states = True
    if not getattr(model.config, 'output_attentions', False):
        model.config.output_attentions = True
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    if hidden_states is None:
        raise ValueError("Hidden states are not returned by the model. Please check the model configuration.")
    if attentions is None:
        raise ValueError("Attentions are not returned by the model. Please check the model configuration.")
    
    # Validate layer_indices
    max_layer = len(hidden_states) - 1
    for idx in layer_indices:
        if idx < 0 or idx > max_layer:
            raise IndexError(f"Layer index {idx} is out of range. The model has {len(hidden_states)} layers.")
    
    # Select and average hidden states from specified layers
    selected_hidden_states = [hidden_states[i] for i in layer_indices]
    mean_hidden_states = [hidden_state.mean(dim=1) for hidden_state in selected_hidden_states]
    
    # Select attentions from specified layers
    selected_attentions = [attentions[i] for i in layer_indices]
    
    return mean_hidden_states, selected_attentions

def visualize_embeddings(embeddings, attentions, sentences, categories):
    fig = make_subplots(rows=2, cols=len(embeddings), 
                        subplot_titles=[f"Hidden States" for i in range(len(embeddings))] +
                                       [f"Layer {i}<br><br><br>Attention" for i in range(len(embeddings))])
    fig.update_layout(title_text="Forward pass through a text embedding model", title_font_size=30)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    all_embeddings = np.concatenate([emb.cpu().numpy() for emb in embeddings])    
    reduced_tsne = tsne.fit_transform(all_embeddings).reshape(len(embeddings), len(sentences), 2)
    
    colors = {'tech': 'red', 'nature': 'blue', 'arts': 'green'}
    
    for i in range(len(embeddings)):
        # Plot hidden states
        for cat in set(categories):
            mask = np.array(categories) == cat
            fig.add_trace(
                go.Scatter(
                    x=reduced_tsne[i, mask, 0],
                    y=reduced_tsne[i, mask, 1],
                    mode='markers',
                    marker=dict(color=colors[cat]),
                    text=[sentences[j] for j in range(len(sentences)) if categories[j] == cat],
                    hoverinfo='text',
                    name=cat
                ),
                row=1, col=i+1
            )
        
        # Plot attention
        attention_matrix = attentions[i].mean(dim=(0, 1)).cpu().numpy()
        fig.add_trace(
            go.Heatmap(
                z=attention_matrix,
                colorscale='Viridis',
                showscale=False
            ),
            row=2, col=i+1
        )
        
        fig.update_xaxes(showticklabels=False, row=1, col=i+1)
        fig.update_yaxes(showticklabels=False, row=1, col=i+1)
        fig.update_xaxes(showticklabels=False, row=2, col=i+1)
        fig.update_yaxes(showticklabels=False, row=2, col=i+1)
    
    fig.update_layout(height=900, width=180*len(embeddings), showlegend=False)
    fig.show()

# Technology
tech_sentences = [
    "Quantum entanglement enables the development of unhackable communication networks.",
    "Neuromorphic engineering aims to replicate the brain's neural structure in microchips.",
    "Atomically precise manufacturing will enable the creation of materials with extraordinary properties.",
    "Quantum machine learning algorithms can efficiently process vast datasets for unparalleled insights.",
    "Brain-computer interfaces may soon allow direct mental control of external devices.",
    "Nanomedicine could revolutionize disease treatment through targeted drug delivery at the molecular level.",
    "Quantum sensors will enable ultra-precise measurements for navigation and scientific discovery.",
    "Photonic integrated circuits use light for faster, more energy-efficient computing than electronic chips.",
    "Metamaterials with engineered structures exhibit optical properties not found in nature.",
    "DNA-based data storage offers immense capacity and long-term stability for archiving information.",
    "Topological insulators conduct electricity on their surface while acting as insulators in their interior.",
    "Quantum key distribution ensures secure communication by detecting eavesdropping attempts.",
    "Neuromorphic sensors mimic biological sensory systems for efficient data processing.",
    "4D printing creates objects that can change shape or self-assemble over time.",
    "Quantum dots are nanoscale semiconductors with tunable optical and electronic properties.",
    "Memristors combine memory and processing capabilities, enabling brain-like computing.",
    "Quantum simulators model complex quantum systems to solve intractable problems.",
    "Optogenetics allows precise control of neural activity using light-sensitive proteins.",
    "Spintronics exploits electron spin for ultra-fast, low-power computing and data storage.",
    "Quantum error correction codes protect quantum information from decoherence and noise."
]

# Nature and Environment
nature_sentences = [
    "The gentle rustling of leaves in the breeze whispers ancient secrets of the forest.",
    "Golden rays of sunlight dance upon the tranquil surface of a mountain lake at dawn.",
    "The majestic silhouette of a soaring eagle embodies the untamed spirit of the wilderness.",
    "Fragrant wildflowers paint the meadow in a vibrant kaleidoscope of color and life.",
    "The thunderous roar of a waterfall echoes through the canyon, a testament to nature's raw power.",
    "A carpet of fallen autumn leaves crunches underfoot, a nostalgic reminder of time's ceaseless march.",
    "The first snowflakes of winter flutter from the heavens, blanketing the landscape in pristine white.",
    "A lone wolf's haunting howl pierces the night, a primal call to the untamed wild.",
    "The delicate unfurling of a fern frond in spring symbolizes nature's endless capacity for renewal.",
    "A shimmering rainbow arcs across the sky after a storm, a fleeting bridge between heaven and earth.",
    "The ancient, gnarled branches of a wise old oak tree reach out like a guardian of the forest.",
    "A shimmering constellation of fireflies illuminates the summer night, a celestial dance on Earth.",
    "The gentle lapping of waves upon the shore sings a lullaby of serenity and inner peace.",
    "A field of golden wheat sways in the sun-kissed breeze, a symbol of nature's bounty.",
    "The graceful silhouette of a lone cypress tree stands sentinel against a fiery sunset sky.",
    "A delicate butterfly alights upon a blossom, a fleeting moment of beauty and symbiosis.",
    "The earthy aroma of petrichor rises from the soil after a cleansing rain, a scent of renewal.",
    "A majestic herd of wild horses gallops across the plain, an emblem of freedom and strength.",
    "The intricate web of a spider glistens with morning dew, a masterpiece of natural engineering.",
    "The soft cooing of a mourning dove echoes through the misty forest at daybreak."
]

# Arts and Culture 
arts_sentences = [
    "Abstract art invites the viewer to project their own meaning onto ambiguous forms and colors.",
    "The haunting melody of a solo violin evokes the depths of the human soul's yearning and sorrow.",
    "Postmodern literature deconstructs traditional narratives, questioning the nature of reality and identity.", 
    "A dancer's fluid movements express the inexpressible, transcending the limitations of language.",
    "Surrealist paintings juxtapose dreamlike imagery, probing the subconscious realms of the psyche.",
    "The minimalist sculpture's stark simplicity provokes contemplation of form, space, and emptiness.", 
    "An epic poem weaves myth and history into a tapestry of a people's collective cultural identity.",
    "Avant-garde theater subverts conventions, shocking audiences out of complacency into self-reflection.",
    "A haiku's brief, evocative lines capture a fleeting moment of enlightenment amidst nature's beauty.", 
    "Atonal music eschews traditional harmony, reflecting the anxiety and alienation of the modern age.",
    "A expressionist painting's distorted forms and lurid colors channel raw, primal emotions.",
    "The improvised riffs of a jazz saxophone solo embody the thrill of spontaneous creation.",  
    "A tragic play probes the depths of human suffering, evoking catharsis through vicarious experience.",
    "The ethereal strains of a choir transport the listener to realms of sacred transcendence.",
    "A Dadaist collage subverts logic and meaning, celebrating absurdity and chance.",
    "The passionate brushstrokes of a Abstract Expressionist painting capture the artist's inner turmoil.",
    "An ironic postmodern novel blurs the line between reality and fiction, challenging assumptions.",  
    "A prima ballerina's graceful pirouettes embody the pursuit of unattainable perfection.",
    "A street mural's vivid imagery and social commentary provoke passersby to reflection and action.", 
    "The dissonant chords of an atonal composition shatter expectations, creating a new musical language."
]

sentences = tech_sentences + nature_sentences + arts_sentences
embeddings, attentions = get_embeddings_across_layers(sentences, np.arange(12))
visualize_embeddings(
    embeddings,
    attentions,
    sentences, 
    ['tech'] * len(tech_sentences) + ['nature'] * len(nature_sentences) + ['arts'] * len(arts_sentences)
)