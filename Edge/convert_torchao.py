import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.export import export, ExportedProgram
from executorch import exir
from executorch.exir import EdgeProgramManager, to_edge
from EMAT_model.EMAT import EMAT
from torchao.quantization import quantize_, int8_dynamic_activation_int4_weight

from torchao.utils import unwrap_tensor_subclass


# Custom KL loss function
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

# Define new model class with mean calculation
class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.win_size = 100  # You need to define this since it's used in your loss computation
        self.criterion = nn.MSELoss(reduction='none')  # Use 'reduction' instead of 'reduce'

    def forward(self, x):
        # Get the original model output
        output, series, prior, sigma = self.original_model(x)

        # Initialize variables
        loss = torch.mean(self.criterion(x, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0

        # Compute series and prior losses
        for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * 50
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * 50
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * 50
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * 50
            # Metric
        # Compute metric and final loss
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy = cri.reshape(-1)  # Reshape as needed

        return attens_energy


def convert(config):
    precision = torch.float32
    device = torch.device("cpu")
    # initial EM-AT model
    model = EMAT(win_size=config.win_size, enc_in=config.input_c, c_out=config.output_c, e_layers=config.e_layer_num)
    
    params = 'e' + str(config.num_epochs) + '_k' + str(config.k) + '_l' + str(config.e_layer_num) + '_b' + str(config.batch_size)
    
    # this path is just an example of the .pth file path
    PATH = "./Cloud/checkpoints/ensemble_bgl/BGL_e3_k3_l3_b32.pth"

    model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))

    # Load the modified model
    new_model = ModifiedModel(model)  # Pass initialized model, not class
    new_model.eval()
    new_model = new_model.to(dtype=precision, device="cpu")

    # only works for torch 2.4+

    # A8W4
    quantize_(new_model, int8_dynamic_activation_int4_weight())
    
    # torch lower than 2.6
    m_unwrapped = unwrap_tensor_subclass(new_model)
    
    sample_inputs = (torch.randn(1, config.win_size, config.data_seq_len, dtype=precision, device=device), )
    # lower to executorch and output to .pte file
    exported_program: ExportedProgram = export(m_unwrapped, sample_inputs)
    edge: EdgeProgramManager = to_edge(exported_program)
    executorch_program = edge.to_executorch()

    output_name = f"./{config.model_save_path}/{config.dataset}_{params}.pte"
    with open(output_name, "wb") as file:    
        file.write(executorch_program.buffer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--e_layer_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--data_seq_len', type=int, default=10) 
    parser.add_argument('--input_c', type=int, default=10)
    parser.add_argument('--output_c', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='BGL')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/qbat_bgl')

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    convert(config)
