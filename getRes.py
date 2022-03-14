import random
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff


data = 'PeMS'

# vis Adj
if data == 'METR':
    h_step = 0
    t_step = 5
    num_nodes = 10
elif data == 'PeMS':
    h_step = 0
    t_step = 1
    num_nodes = 10
else:
    h_step = 0
    t_step = 1
    num_nodes = 2

adj = np.load(f'./Datasets/{data}/dists.npy')
enc_adj = np.load(f'./Datasets/{data}/{data}_enc_adj.npy')
dec_adj = np.load(f'./Datasets/{data}/{data}_dec_adj.npy')

adj = np.around(adj, decimals=3)
enc_adj = np.around(enc_adj, decimals=3)
dec_adj = np.around(dec_adj, decimals=3)

rand_node = random.choice(range(adj.shape[0] - num_nodes))

x = list(range(rand_node, rand_node + num_nodes))
y = list(range(rand_node, rand_node + num_nodes))

fig_adj = ff.create_annotated_heatmap(
    z=adj[rand_node: rand_node + num_nodes, rand_node: rand_node + num_nodes],
    x=x,
    y=y,
    colorscale='YlGnBu'
)

# fig_adj = go.Figure(data=go.Heatmap(z=adj[rand_node: rand_node + 4, rand_node: rand_node + 4]))
fig_adj.update_layout(template='plotly_white',
                      margin=dict(l=0, r=0, t=0, b=0),
                      width=700,
                      height=600)

fig_enc_adj_h = ff.create_annotated_heatmap(
    z=enc_adj[h_step, rand_node: rand_node + num_nodes, rand_node: rand_node + num_nodes],
    x=x,
    y=y,
    colorscale='YlGnBu'
)

# fig_enc_adj_h = go.Figure(data=go.Heatmap(z=enc_adj[h_step, rand_node: rand_node + 4, rand_node: rand_node + 4]))
fig_enc_adj_h.update_layout(template='plotly_white',
                            margin=dict(l=0, r=0, t=0, b=0),
                            width=700,
                            height=600)

fig_dec_adj_h = ff.create_annotated_heatmap(
    z=dec_adj[h_step, rand_node: rand_node + num_nodes, rand_node: rand_node + num_nodes],
    x=x,
    y=y,
    colorscale='YlGnBu'
)

# fig_dec_adj_h = go.Figure(data=go.Heatmap(z=dec_adj[h_step, rand_node: rand_node + 4, rand_node: rand_node + 4]))
fig_dec_adj_h.update_layout(template='plotly_white',
                            margin=dict(l=0, r=0, t=0, b=0),
                            width=700,
                            height=600)

fig_enc_adj_t = ff.create_annotated_heatmap(
    z=enc_adj[t_step, rand_node: rand_node + num_nodes, rand_node: rand_node + num_nodes],
    x=x,
    y=y,
    colorscale='YlGnBu'
)

# fig_enc_adj_t = go.Figure(data=go.Heatmap(z=enc_adj[t_step, rand_node: rand_node + 4, rand_node: rand_node + 4]))
fig_enc_adj_t.update_layout(template='plotly_white',
                            margin=dict(l=0, r=0, t=0, b=0),
                            width=700,
                            height=600)

fig_dec_adj_t = ff.create_annotated_heatmap(
    z=dec_adj[t_step, rand_node: rand_node + num_nodes, rand_node: rand_node + num_nodes],
    x=x,
    y=y,
    colorscale='YlGnBu'
)

# fig_dec_adj_t = go.Figure(data=go.Heatmap(z=dec_adj[t_step, rand_node: rand_node + 4, rand_node: rand_node + 4]))
fig_dec_adj_t.update_layout(template='plotly_white',
                            margin=dict(l=0, r=0, t=0, b=0),
                            width=700,
                            height=600)

fig_adj.show()
fig_enc_adj_h.show()
fig_dec_adj_h.show()
fig_enc_adj_t.show()
fig_dec_adj_t.show()

fig_adj.write_image(f'./Fig/{data}_adj.pdf')
fig_enc_adj_h.write_image(f'./Fig/{data}_{h_step}step_enc_adj.pdf')
fig_dec_adj_h.write_image(f'./Fig/{data}_{h_step}step_dec_adj.pdf')
fig_enc_adj_t.write_image(f'./Fig/{data}_{t_step}step_enc_adj.pdf')
fig_dec_adj_t.write_image(f'./Fig/{data}_{t_step}step_dec_adj.pdf')

# vis Flow
p = np.load(f'./Datasets/{data}/{data}_pred.npy')
t = np.load(f'./Datasets/{data}/{data}_true.npy')
gap = 288
sample_times = 2

for st in range(sample_times):
    node = random.choice(range(p.shape[1]))
    start = random.choice(range(p.shape[0] - gap))

    fig = go.Figure()

    p_line = go.Scatter(y=p[start: start + gap, node], mode='lines', name='GDFormer',
                        line=dict(color='#D9436B'))
    t_line = go.Scatter(y=t[start: start + gap, node], mode='lines', name=f'Ground Truth of {data}',
                        line=dict(color='#013440'))

    fig.add_trace(p_line)
    fig.add_trace(t_line)

    fig.update_xaxes(showline=True, linecolor='black')
    fig.update_yaxes(showline=True, linecolor='black')
    fig.update_layout(template='plotly_white',
                      xaxis_title='Time',
                      yaxis_title='Traffic Volume',
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,
                      font=dict(
                          family="Times New Roman",
                          size=24,
                      ),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      ))

    fig.show()
    fig.write_image(f'./Fig/{data}_{node}th_node_start_at{start}_flow.pdf')
