import torch
import torchvision


class ResNetBackbone(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        builders = {
            "18": torchvision.models.resnet18,
            "34": torchvision.models.resnet34,
            "50": torchvision.models.resnet50,
        }
        model = builders[layers](weights=None)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class ParsingNet(torch.nn.Module):
    def __init__(self, backbone, num_grid_row, num_cls_row, num_grid_col, num_cls_col,
                 num_lane_on_row, num_lane_on_col, input_height, input_width):
        super().__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.input_height = input_height
        self.input_width = input_width

        self.dim1 = num_grid_row * num_cls_row
        self.dim2 = 2 * num_cls_row
        self.dim3 = num_grid_col * num_cls_col
        self.dim4 = 2 * num_cls_col
        self.input_dim = (input_height // 32) * (input_width // 32) * 9
        mlp_mid_dim = 2048

        self.model = ResNetBackbone(backbone)
        self.cls_distribute = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 20, 3, padding=1),
        )
        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
        )
        self.cls_row = torch.nn.Linear(mlp_mid_dim, self.dim1 + self.dim2)
        self.cls_col = torch.nn.Linear(mlp_mid_dim, self.dim3 + self.dim4)
        self.pool = torch.nn.Conv2d(512, 8, 1)

    def forward(self, x):
        _, _, fea = self.model(x)
        h, w = self.input_height // 32, self.input_width // 32

        lane_token = self.cls_distribute(fea).reshape(-1, 20, 1, h, w)
        fea = self.pool(fea).unsqueeze(1).repeat(1, 20, 1, 1, 1)
        fea = torch.cat([fea, lane_token], 2).view(-1, self.input_dim)

        out = self.cls(fea).reshape(-1, 20, 2048)
        out_row = self.cls_row(out[:, :10, :]).permute(0, 2, 1)
        out_col = self.cls_col(out[:, 10:, :]).permute(0, 2, 1)

        return {
            "loc_row": out_row[:, :self.dim1, :].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
            "loc_col": out_col[:, :self.dim3, :].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            "exist_row": out_row[:, self.dim1:self.dim1 + self.dim2, :].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
            "exist_col": out_col[:, self.dim3:self.dim3 + self.dim4, :].view(-1, 2, self.num_cls_col, self.num_lane_on_col),
        }


def build_model(cfg, device):
    net = ParsingNet(
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
    )
    return net.to(device)


def load_checkpoint(net, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(cleaned, strict=False)
