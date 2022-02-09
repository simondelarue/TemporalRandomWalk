# Temporal Random Walk

Considering a temporal network, we implement a framework that computes **temporal random walk** according to some sampling strategy.

**Temporal network**  

A temporal network <img src="https://render.githubusercontent.com/render/math?math=G=(V, E^T, T)"> is a set of nodes <img src="https://render.githubusercontent.com/render/math?math=V"> and timestamped edges <img src="https://render.githubusercontent.com/render/math?math=E^T"> that exist within a period of time <img src="https://render.githubusercontent.com/render/math?math=T">. This graph can be considered as a sequence of interactions, where each interaction is an event that comes in the form of a triplet <img src="https://render.githubusercontent.com/render/math?math=(t, u, v)">, where <img src="https://render.githubusercontent.com/render/math?math=u"> and <img src="https://render.githubusercontent.com/render/math?math=v"> represent nodes in <img src="https://render.githubusercontent.com/render/math?math=G"> and <img src="https://render.githubusercontent.com/render/math?math=t"> is the time at which these nodes interacted with each other.  
Temporal networks are encoded as **dictionaries**, where each key is a node in the graph and each corresponding value is a dictionary that maps for each timestep, a list of nodes which interacted with the key.

**Temporal random walk**  

A temporal random walk is defined as a sequence of nodes in a temporal network. Such a walk exists between two nodes, if one can find a sequence of edges with non-decreasing timesteps that connects these two nodes.


## Table of contents

1. [Setup](#Setup)  
2. [Usage](#Usage)  
3. [Datasets](#Datasets)  


## 1. Setup <a class="anchor" id="Setup"></a>

### Install dependencies
```bash
[~/]$ git clone https://github.com/simondelarue/TemporalRandomWalk.git
[~/]$ cd TemporalRandomWalk
[~/TemporalRandomWalk]$ python -m venv myvenv
[~/TemporalRandomWalk]$ source myvenv/bin/activate
(myvenv)[~/TemporalRandomWalk]$ pip install requirements.txt
```

## 2. Usage <a class="anchor" id="Usage"></a>  

Run `python src/main.py` using the following arguments to chose data, size of test graph or sampling strategy:

**Arguments**
``` system
--data              Data source {SF2H, HighSchool, ia-contact, ia-contacts_hypertext2009, fb-forum, ia-enron-employees}
--test_size        Size of test dataset (in proportion of total number of edges)
--strategy          Sampling strategy: \{linear, exponential, uniform\}    
```

**Example**
``` bash
$ python src/main.py --data SF2H --test_size 0.1 --strategy linear

Device : cpu
Preprocessing data ...
Splitting data ...

Creating stream graphs ...

Train graph
# nodes : 403
# edges : 133874

Test graph
# nodes : 201
# edges : 6648

Start edge: (306, 87, 145080)

Temporal random walk: [306, 87, 306, 87, 306, 87, 199, 79, 199, 79, 199, 79, 133, 165, 79, 199, 79, 199, 79, 133]
```

## 3. Datasets <a class="anchor" id="Datasets"></a>

In this work we use real-world datasets, where interactions between entities are encoded through *events* in the form of triplets <img src="https://render.githubusercontent.com/render/math?math=(t, uv)">, where <img src="https://render.githubusercontent.com/render/math?math=u"> and <img src="https://render.githubusercontent.com/render/math?math=v"> represent nodes in the graph, and <img src="https://render.githubusercontent.com/render/math?math=t"> is the time at which these nodes interacted with each other. We use the following datasets:

| Dataset      |
|--------------|
| [```SF2H```](http://www.sociopatterns.org/datasets/sfhh-conference-data-set/) |
| [```HighSchool```](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) |  
| [```ia-contact```](https://networkrepository.com/ia-contact.php) | 
| [```ia-contacts-hypertext2009```](http://www.sociopatterns.org/datasets/hypertext-2009-dynamic-contact-network/) |
| [```ia-enron-employees```](https://networkrepository.com/ia_enron_employees.php) |