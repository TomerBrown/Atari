# Atari

To run the code using epsilon greedy, run:

`python main.py`

To run using softmax action with beta_t=log(t), run:

`python main.py --action_selection softmax`

To change the beta_t function, choose between:

`python main.py --action_selection softmax --beta_func log`

`python main.py --action_selection softmax --beta_func loglog`

`python main.py --action_selection softmax --beta_func root --beta_pow 0.3`

To train using Adam optimization, use parameter:

`python main.py --optim Adam`

To execute different runs without overwriting previous files, add the parameter: `--run_name "my run name"`

saved statistics pkl will have the prefix of your run name, and the output folder under tmp will have add the run name as suffix.

If `--run_name "my run name"` is not used, it will default to `unnamed`