<h1>GraphNCF-A: Graph Neural Collaborative Filtering with Attention</h1>

<p>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.12+-orange.svg" alt="PyTorch">
  </a>
</p>

<p>
  <strong>GraphNCF-A</strong> is a recommendation model that extends <strong>Neural Graph Collaborative Filtering (NGCF)</strong> by incorporating a multi-head <strong>attention</strong> mechanism. It leverages the user-item interaction graph to learn embeddings for users and items and uses <strong>self-attention on graph neighbors</strong> to adaptively weight important interactions. The model is trained with a <strong>Bayesian Personalized Ranking (BPR)</strong> loss for implicit feedback, aiming to improve top-K recommendation accuracy (Hit Rate and NDCG).
</p>

<hr>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#installation-and-setup">Installation and Setup</a></li>
  <li><a href="#dataset-preparation">Dataset Preparation</a></li>
  <li><a href="#running-the-model">Running the Model</a></li>
  <li><a href="#model-architecture">Model Architecture</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#project-structure">Project Structure</a></li>
  <li><a href="#references">References</a></li>
</ul>

<hr>

<h2 id="installation-and-setup">Installation and Setup</h2>
<h3>Requirements</h3>
<ul>
  <li>Python 3.8 or higher</li>
  <li>PyTorch 1.12 or higher</li>
  <li><code>numpy</code>, <code>pandas</code>, <code>scipy</code></li>
</ul>

<p>Install the required packages via pip:</p>
<pre><code>pip install torch numpy pandas scipy</code></pre>

<hr>

<h2 id="dataset-preparation">Dataset Preparation</h2>
<p>The code supports the <strong>MovieLens</strong> dataset in various sizes (100K, 1M, 10M, 20M). You can either let the code download the dataset automatically or provide the files manually:</p>
<ul>
  <li>
    <strong>Automatic Download:</strong> Use the <code>-dl True</code> flag when running the program (handled by the <code>Download</code> class in <code>Utils.py</code>).
  </li>
  <li>
    <strong>Manual Setup:</strong> Download the dataset from the <a href="https://grouplens.org/datasets/movielens/">GroupLens website</a> and place it into a <code>dataset</code> directory. Ensure the file has the correct structure (with <code>userId</code> and <code>movieId</code> columns).
  </li>
</ul>

<hr>

<h2 id="running-the-model">Running the Model</h2>
<p>After setting up the environment and dataset, run the model via the command line using <code>Main.py</code>. For example:</p>
<pre><code>python3 Main.py --epoch 10 --batch 256 --lr 1e-3 --download True --top_k 10 --num_heads 4 --file_size 100k</code></pre>
<p>This command will:</p>
<ul>
  <li>Train the model for 10 epochs (<code>--epoch 10</code>).</li>
  <li>Use a batch size of 256 (<code>--batch 256</code>).</li>
  <li>Set the learning rate to 1e-3 (<code>--lr 1e-3</code>).</li>
  <li>Download the dataset if not available (<code>--download True</code>).</li>
  <li>Evaluate using top-10 recommendations (<code>--top_k 10</code>).</li>
  <li>Use 4 attention heads (<code>--num_heads 4</code>).</li>
  <li>Work on the 100K version of MovieLens (<code>--file_size 100k</code>).</li>
</ul>

<hr>

<h2 id="model-architecture">Model Architecture</h2>
<p>GraphNCF-A builds upon the NGCF framework by incorporating a multi-head attention mechanism. The key components include:</p>

<h3>Graph Construction</h3>
<ul>
  <li><strong>Interaction Graph:</strong> User-item interactions are modeled as a bipartite graph.</li>
  <li><strong>Adjacency Matrix:</strong> The <code>Laplacian</code> class in <code>Laplacian_mat.py</code> constructs a sparse adjacency matrix that represents the interactions. Self-connections are added via an identity matrix.</li>
  <li><strong>Normalization:</strong> The normalized Laplacian, computed as <em>\( \tilde{A} = D^{-1/2} A D^{-1/2} \)</em>, scales aggregation from high-degree nodes.
  </li>
</ul>

<h3>Embedding Initialization</h3>
<ul>
  <li><strong>User &amp; Item Embeddings:</strong> Each user and item is assigned a trainable embedding vector (e.g., 64 dimensions).</li>
  <li><strong>Initialization:</strong> Xavier uniform initialization is applied to both embeddings (see <code>GraphNCF_A.py</code>).</li>
</ul>

<h3>Graph Propagation with Attention</h3>
<p>GraphNCF-A refines embeddings through multiple propagation layers:</p>
<ul>
  <li><strong>Neighbor Aggregation:</strong> The model aggregates neighbor embeddings using sparse matrix multiplication with the normalized Laplacian.</li>
  <li><strong>Attention Mechanism:</strong> A multi-head attention layer (defined in <code>MultiHeadAttentionLayer</code> in <code>GraphNCF_A.py</code>) computes attention weights for each edge, enabling the model to focus on informative neighbors.</li>
  <li><strong>Second-Order Interaction:</strong> In addition to aggregation, an element-wise product between node embeddings and their neighbors is computed, passed through attention, and combined.</li>
  <li><strong>Activation &amp; Normalization:</strong> LeakyReLU activation, dropout, and L2 normalization are applied.</li>
  <li><strong>Concatenation:</strong> Embeddings from all layers (including the initial layer) are concatenated to form final representations.</li>
</ul>

<details>
  <summary><strong>Detailed Pseudocode for a Propagation Layer</strong></summary>
  <pre><code>
for layer in range(L): 
    # First message: Neighbor aggregation
    agg_neighbors = sparse_norm_adj @ prev_embedding
    agg_neighbors = attention_layer(agg_neighbors, edge_index)
    agg_neighbors = Linear_W1[layer](agg_neighbors)

    # Second message: Self-neighbor interaction (element-wise product)
    self_neighbor = prev_embedding * prev_embedding
    self_neighbor = sparse_norm_adj @ self_neighbor
    self_neighbor = attention_layer(self_neighbor, edge_index)
    self_neighbor = Linear_W2[layer](self_neighbor)

    # Combine and apply activation
    msg = agg_neighbors + self_neighbor 
    prev_embedding = LeakyReLU(msg)
    prev_embedding = Dropout(prev_embedding)
    prev_embedding = normalize(prev_embedding, p=2)
    all_layer_embeddings.append(prev_embedding)
  </code></pre>
</details>

<h3>Prediction and Training Objective</h3>
<ul>
  <li><strong>Scoring:</strong> The affinity between a user and an item is computed as the dot product of their final embeddings.</li>
  <li><strong>BPR Loss:</strong> The model is trained using Bayesian Personalized Ranking (BPR) loss (implemented in <code>BPR_Loss.py</code>), encouraging positive items to be ranked higher than negatives.</li>
  <li><strong>Negative Sampling:</strong> For each positive interaction, negative items are sampled (handled in the <code>MovieLens</code> dataset class in <code>Utils.py</code>).</li>
</ul>

<hr>

<h2 id="results">Results</h2>
<p>Below is an example of evaluation metrics (Hit Rate and NDCG) for various configurations on MovieLens datasets:</p>

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Best NDCG@10</th>
      <th>HR@10</th>
      <th># Layers</th>
      <th>Epochs</th>
      <th>Batch Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MovieLens 100K</td>
      <td>0.5900</td>
      <td>0.8250</td>
      <td>3</td>
      <td>20</td>
      <td>256</td>
    </tr>
    <tr>
      <td>MovieLens 100K</td>
      <td>0.5750</td>
      <td>0.8320</td>
      <td>4</td>
      <td>20</td>
      <td>256</td>
    </tr>
    <tr>
      <td>MovieLens 100K</td>
      <td>0.5600</td>
      <td>0.8450</td>
      <td>5</td>
      <td>20</td>
      <td>256</td>
    </tr>
    <tr>
      <td>MovieLens 1M</td>
      <td>0.5050</td>
      <td>0.7650</td>
      <td>3</td>
      <td>20</td>
      <td>256</td>
    </tr>
    <tr>
      <td>MovieLens 1M</td>
      <td>0.5000</td>
      <td>0.7720</td>
      <td>4</td>
      <td>20</td>
      <td>256</td>
    </tr>
    <tr>
      <td>MovieLens 1M</td>
      <td>0.4900</td>
      <td>0.7600</td>
      <td>5</td>
      <td>20</td>
      <td>256</td>
    </tr>
  </tbody>
</table>

<p><em>Note:</em> These numbers are indicative. Adjust hyperparameters based on your dataset and experimental setup.</p>

<hr>

<h2 id="project-structure">Project Structure</h2>
<pre>
├── GraphNCF_A.py        # Contains the GraphNCF-A model and multi-head attention implementation.
├── BPR_Loss.py          # Implements the BPR loss function with L1/L2 regularization.
├── Laplacian_mat.py     # Constructs the interaction and normalized Laplacian matrices.
├── Utils.py             # Utilities for downloading and processing the MovieLens dataset.
├── Train.py             # Training loop that handles forward/backward passes and model updates.
├── Evaluation.py        # Evaluation metrics: Hit Rate and NDCG for top-K recommendations.
├── Parser.py            # Command-line argument parser for configurable hyperparameters.
├── Main.py              # Entry point: sets up data, builds the model, and starts training.
└── Readme.docx          # Original project documentation (key instructions integrated here).
</pre>

<hr>

<h2 id="references">References</h2>
<ol>
  <li>
    <strong>Neural Graph Collaborative Filtering</strong><br>
    Xiang Wang <em>et al.</em>, SIGIR 2019.<br>
    <a href="https://arxiv.org/abs/1905.08108">arXiv:1905.08108</a>
  </li>
  <li>
    <strong>Neural Collaborative Filtering</strong><br>
    Xiangnan He <em>et al.</em>, WWW 2017.<br>
    <a href="https://arxiv.org/abs/1708.05031">arXiv:1708.05031</a>
  </li>
  <li>
    <strong>Graph Attention Networks</strong><br>
    Petar Veličković <em>et al.</em>, ICLR 2018.<br>
    <a href="https://arxiv.org/abs/1710.10903">arXiv:1710.10903</a>
  </li>
</ol>
