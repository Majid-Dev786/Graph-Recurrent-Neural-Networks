# First, I'm bringing in all the necessary tools. NumPy helps me with the heavy lifting of matrix operations,
# Plotly Graph Objects for some snazzy visualizations, and NetworkX for dealing with graph structures.
import numpy as np
import plotly.graph_objects as go
import networkx as nx

# Here's my GraphRNN class, a recurrent neural network tailored for graph data.
class GraphRNN:
    def __init__(self, num_nodes, num_features, hidden_units, num_classes):
        # Storing the basic setup: how many nodes and features we're dealing with, the size of the hidden layer, and the number of classes for the output.
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_classes = num_classes

        # Initializing the weight matrices for the hidden and output layers with random values.
        # This randomness is a good starting point for the learning process.
        self.W_h = np.random.randn(num_features, hidden_units)
        self.W_y = np.random.randn(hidden_units, num_classes)

    def forward(self, adj_matrix, node_features):
        # Preparing a container for the hidden states, starting them all off as zeroes.
        hidden_state = np.zeros((self.num_nodes, self.hidden_units))
        # This loop calculates the hidden state for each node in the graph, one at a time.
        for t in range(self.num_nodes):
            hidden_state[t] = np.dot(node_features[t], self.W_h) + np.dot(adj_matrix[t], hidden_state)

        # The output is calculated by applying the weights to the hidden states.
        output = np.dot(hidden_state, self.W_y)
        return output

    def train(self, adj_matrix, node_features, labels, learning_rate, num_epochs):
        # Preparing a blank slate for the hidden states, just like in the forward pass.
        hidden_state = np.zeros((self.num_nodes, self.hidden_units))
        # Time to learn! This loop runs through the training process for a specified number of epochs.
        for epoch in range(num_epochs):
            Y_pred = self.forward(adj_matrix, node_features)
            # Calculating the loss as the mean squared error between the predictions and actual labels.
            loss = np.mean((Y_pred - labels) ** 2)

            # Preparing to adjust the weights based on the error, starting from zero.
            dW_h = np.zeros_like(self.W_h)
            dW_y = np.zeros_like(self.W_y)

            # This loop calculates how much we need to adjust each weight.
            for t in range(self.num_nodes):
                dW_y += np.outer(hidden_state[t], (Y_pred[t] - labels[t]))
                dW_h += np.dot(node_features[t].reshape(-1, 1), hidden_state[t].reshape(1, -1))

            # Adjusting the weights in the opposite direction of the gradient to reduce the error.
            self.W_h -= learning_rate * dW_h
            self.W_y -= learning_rate * dW_y

            # Every ten epochs, it's good to check in and see how the training is going.
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')

# A simple dataset class to generate some data to work with.
class Dataset:
    def create_sample(self):
        # Constructing a small, sample adjacency matrix and corresponding node features and labels.
        adj_matrix = np.array([[0, 1, 0, 1],
                               [1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [1, 0, 1, 0]])

        node_features = np.array([[1, 0],
                                  [0, 1],
                                  [1, 0],
                                  [0, 1]])

        labels = np.array([[0.5, 0.5],
                           [0.2, 0.8],
                           [0.7, 0.3],
                           [0.9, 0.1]])

        return adj_matrix, node_features, labels

# And here's how we can see what we're doing, with a class for visualizing the graph.
class GraphVisualizer:
    def visualize_graph(self, adj_matrix):
        # First, turning our adjacency matrix into a NetworkX graph.
        G = nx.from_numpy_array(adj_matrix)
        # Finding a nice layout to visualize our graph.
        pos = nx.spring_layout(G)
        # Setting up the traces for plotting edges.
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # And now for the nodes, making sure each one is visible and properly labeled.
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        # Putting it all together into a figure and showing it.
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Graph Visualization',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        fig.show()

# The main event! This is where we put it all into action.
def main():
    dataset = Dataset()
    graph_visualizer = GraphVisualizer()
    adj_matrix, node_features, labels = dataset.create_sample()

    # Creating an instance of our GraphRNN with the sample data dimensions.
    grnn = GraphRNN(num_nodes=adj_matrix.shape[0], num_features=node_features.shape[1],
                    hidden_units=32, num_classes=labels.shape[1])

    learning_rate = 1
    num_epochs = 100

    # Training our neural network with the generated data.
    grnn.train(adj_matrix, node_features, labels, learning_rate, num_epochs)
    Y_pred = grnn.forward(adj_matrix, node_features)
    # After training, we calculate and print the mean squared error of our predictions.
    mse = np.mean((Y_pred - labels) ** 2)
    print(f'Predicted output:\n{Y_pred}')
    print(f'Mean Squared Error: {mse}')
    # And finally, visualizing the graph to see what we've been working with.
    graph_visualizer.visualize_graph(adj_matrix)

# Making sure our script executes the main function when run.
if __name__ == "__main__":
    main()