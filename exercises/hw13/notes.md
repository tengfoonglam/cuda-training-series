# Lesson 13 - CUDA Graphs

## CUDA Graphs Introduction

#### Definition of a CUDA Graph
- A graph node is any asynchronous CUDA operation: A sequence of operations, connected by dependencies
- Operations include:
  - Kernel launch: GPU kernel
  - CPU function call: CPU callbacks
  - Memcopy / memset: GPU data management
  - Memory alloc / free: Inline memory allocation
  - Sub-graphs: Graphs are heirarchical

#### Features
- Graphs can be generated once then launched repeatedly
- Graph launch submits all work at once, reducing CPU costs, as compared to launching multiple kernels separately

#### Three-Stage Execution Model
1. **Define** - Single graph "template": created in host code/loaded from disk/built up from libraries
2. **Instantiate** - Multiple "executable graphs":
    - snapshot of template
    - sets up and initialises GPU execution structures
    - create once and run many times
3. **Execute** - Executbale graphs running in CUDA streams: Concurrency in graph is not limited by stream

#### Modifying Graphs In-Place
|                      | Parameters     | Kernel Topology |
| ---------------------| -------------- | --------------- |
| Normal Stream Launch | May change     | May change      |
| Graph Update         | May change     | May not change  |
| Graph Re-Launch      | May not change | May not change  |

#### Additional Points
- User must define execution location (e.g. which GPU for multiple-gpu devices) for each node
- Edges represent execution dependencies and not data dependencies

## Methods of Programming CUDA Graphs

#### Stream Capture
- Records operations without actually launching a kernel
- All operations need to be asynchronous any any calls to `cudaStreamSynchronize()` or any other synchronous operation will cause the capture operation to fail
- Also records kernel calls by external library functions
- `cudaEventRecord` used to record inter-stream dependencies and creates forks in the graph

Example
```C++
// Start by initiating stream capture
cudaStreamBeginCapture(&stream1);
// Build stream work as usual
A<<< ..., stream1 >>>();
cudaEventRecord(e1, stream1);
B<<< ..., stream1 >>>();
cudaStreamWaitEvent(stream2, e1);
C<<< ..., stream2 >>>();
cudaEventRecord(e2, stream2);
cudaStreamWaitEvent(stream1, e2);
D<<< ..., stream1 >>>();
// Now convert the stream to a graph
cudaStreamEndCapture(stream1, &graph);
```

#### Create Graphs Directly
- Refer to CUDA API for documentation

Example
```C++
// Define graph of work + dependencies
cudaGraphCreate(&graph);
cudaGraphAddNode(graph, kernel_a, {}, ...);
cudaGraphAddNode(graph, kernel_b, { kernel_a }, ...);
cudaGraphAddNode(graph, kernel_c, { kernel_a }, ...);
cudaGraphAddNode(graph, kernel_d, { kernel_b, kernel_c }, ...);
// Instantiate graph and apply optimizations
cudaGraphInstantiate(&instance, graph);
// Launch executable graph 100 times
for(int i=0; i<100; i++)
{
  cudaGraphLaunch(instance, stream);
}
```

#### Combining Graph & Stream Work

Example - Adding captured graph into an existing graph
```C++
// Create root node of graph via explicit API
cudaGraphAddNode(main_graph, X, {}, ...);
// Capture the library call into a subgraph
cudaStreamBeginCapture(&stream);
libraryCall(stream); // Launches A, B, C, D
cudaStreamEndCapture(stream, &library_graph);
// Insert the subgraph into main_graph as node “Y”
cudaGraphAddChildGraphNode(Y, main_graph, { X } ... library_graph);
// Continue building main graph via explicit API
cudaGraphAddNode(main_graph, Z, { Y }, ...);
```

#### Inserting Non-Graph CUDA Work between Graphs
- Possible as long as you can run the CUDA work in a stream
- Note: stream used to launch the kernels/graphs only used to order the work, work might still run on different streams than the launch stream

```C++
launchWork(cudaGraphExec_t i1, cudaGraphExec_t i2,
 CPU_Func cpu, cudaStream_t stream) {
 A <<< 256, 256, 0, stream >>>(); // Kernel launch
 cudaGraphLaunch(i1, stream); // Graph launch
 cudaStreamAddCallback(stream, cpu); // CPU callback
 cudaGraphLaunch(i2, stream); // Graph launch
 cudaStreamSynchronize(stream);
}
```

## CUDA Graphs Benefits Summary
- Rapid re-issue of work - Graphs can be generated once and executed repeatedly
- Graph nodes include GPU work, CPU work and data movement (i.e. heterogeneous node types)
- Can optimise for both multi-device and heterogeneous dependencies
