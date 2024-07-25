import torch

class FaultLocalizationMetrics:
    @staticmethod
    def calculate_ochiai(Ef, Ep, Nf, Np):
        numerator = Ef
        denominator = torch.sqrt((Ef + Nf) * (Ef + Ep))
        return torch.div(numerator, denominator)

    @staticmethod
    def calculate_tarantula(Ef, Ep, Nf, Np):
        numerator = torch.div(Ef, Ef + Nf)
        denominator = numerator + torch.div(Ep, Ep + Np)
        return torch.div(numerator, denominator)

    @staticmethod
    def calculate_zoltar(Ef, Ep, Nf, Np):
        return torch.div(Ef, Ef + Nf + Ep + torch.div(10000 * Nf * Ep, Ef))

    @staticmethod
    def calculate_wong1(Ef, Ep, Nf, Np):
        return Ef - Ep


class RelevanceScore:
    def __init__(self, device='cuda'):
        self.device = device
        self.Ep = None
        self.Ef = None
        self.Np = None
        self.Nf = None
        self.scores_dict = {}

    def calculate_relevance_scores(self, sampled_tensor, mask, N):
        sampled_tensor = sampled_tensor.to(self.device)

        _, C, H, W = sampled_tensor.shape
        all_indices = torch.arange(N, device=self.device)
        pass_indices = all_indices[all_indices % 2 == (0 if N % 2 == 0 else 1)]
        fail_indices = all_indices[all_indices % 2 != (0 if N % 2 == 0 else 1)]

        shape = (N, 1, H, W)
        tensor_ones = torch.ones(shape).to(self.device)

        executed_tensors = mask
        not_executed_tensors = tensor_ones - mask

        e_pass_tensors = executed_tensors[pass_indices]
        e_fail_tensors = executed_tensors[fail_indices]
        n_pass_tensors = not_executed_tensors[pass_indices]
        n_fail_tensors = not_executed_tensors[fail_indices]

        self.Ep = e_pass_tensors.sum(dim=0)
        self.Ef = e_fail_tensors.sum(dim=0)
        self.Np = n_pass_tensors.sum(dim=0)
        self.Nf = n_fail_tensors.sum(dim=0)

    def calculate_all_scores(self, img, masks, N):
        self.calculate_relevance_scores(img, masks, N)

        ochiai_scores = FaultLocalizationMetrics.calculate_ochiai(self.Ef, self.Ep, self.Nf, self.Np)
        tarantula_scores = FaultLocalizationMetrics.calculate_tarantula(self.Ef, self.Ep, self.Nf, self.Np)
        zoltar_scores = FaultLocalizationMetrics.calculate_zoltar(self.Ef, self.Ep, self.Nf, self.Np)
        wong1_scores = FaultLocalizationMetrics.calculate_wong1(self.Ef, self.Ep, self.Nf, self.Np)

        self.scores_dict = {
            'Ep': self.Ep,
            'Ef': self.Ef,
            'Np': self.Np,
            'Nf': self.Nf,
            'ochiai': ochiai_scores,
            'tarantula': tarantula_scores,
            'zoltar': zoltar_scores,
            'wong1': wong1_scores
        }

        return self.scores_dict

    def create_pixel_dataset(self, img_shape):
        H, W = img_shape[-2:]
        dataset = []
        for i in range(H):
            for j in range(W):
                pixel_data = {
                    'position': (i, j),
                    'Ep': self.scores_dict['Ep'][0, i, j].item(),
                    'Ef': self.scores_dict['Ef'][0, i, j].item(),
                    'Np': self.scores_dict['Np'][0, i, j].item(),
                    'Nf': self.scores_dict['Nf'][0, i, j].item(),
                    'ochiai': self.scores_dict['ochiai'][0, i, j].item(),
                    'tarantula': self.scores_dict['tarantula'][0, i, j].item(),
                    'zoltar': self.scores_dict['zoltar'][0, i, j].item(),
                    'wong1': self.scores_dict['wong1'][0, i, j].item()
                }
                dataset.append(pixel_data)
        return dataset

    def run(self, img, masks, N):
        self.calculate_all_scores(img, masks, N)
        dataset = self.create_pixel_dataset(img.shape)
        return dataset


# Usage example:
img = torch.randn((10, 3, 32, 32))  # Example tensor, replace with actual image tensor
masks = torch.randn((10, 1, 32, 32))  # Example masks, replace with actual masks
N = 10  # Number of samples

relevance_score_calculator = RelevanceScore(device='cuda')
pixel_dataset = relevance_score_calculator.run(img, masks, N)

# 'pixel_dataset' now contains the dataset with scores for each pixel.
