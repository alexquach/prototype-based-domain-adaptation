
## Protoype-based Domain Adaptation
Extends the paper: 
`Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions`: https://arxiv.org/abs/1710.04806

## Important files:
### Models:
/src/models/proto_model.py
* Implements a single ProtoModel similar to the paper above
* 
/src/models/cycle_model.py
* Implements a pair of ProtoModels (one for the source and one for the target domains) to learn a transition function that maps from the source domain to the target domain

### Training Scripts:
/src/scripts/train.py
* Trains a single ProtoModel

/src/scripts/train_cycle.py
* Trains a Cycle Model
