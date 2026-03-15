from model.nas.gencoder import GEncoder
from model.nas.archgen import AG
from model.nas.supernet import Network

from autogllight.nas.space import BaseSpace
from autogllight.nas.space.graces_space.genotypes import NA_PRIMITIVES, LA_PRIMITIVES, POOL_PRIMITIVES, READOUT_PRIMITIVES, ACT_PRIMITIVES

# Model used for Graces training
class GraceModel(BaseSpace):
    def __init__(self, input_dim, mol, virtual, args, use_forward):
        super().__init__()
        self.input_dim = input_dim
        self.mol = mol
        self.virtual = virtual
        self.args = args
        self.use_forward = use_forward
        self.build_graph()

    def build_graph(self):
        self.supernet0 = GEncoder(
            in_dim=self.input_dim,
            hidden_size=self.args.graph_dim,
            num_layers=2,
            dropout=0.5,
            epsilon=self.args.epsilon,
            args=self.args,
            with_conv_linear=self.args.with_conv_linear,
            mol=self.mol,
            virtual=self.virtual,
        )
        self.supernet = Network(
            in_dim=self.input_dim,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            epsilon=self.args.epsilon,
            args=self.args,
            with_conv_linear=self.args.with_conv_linear,
            mol=self.mol,
            virtual=self.virtual,
        )
        num_na_ops = len(NA_PRIMITIVES)
        num_pool_ops = len(POOL_PRIMITIVES)
        self.ag = AG(args=self.args, num_op=num_na_ops, num_pool=num_pool_ops)
        self.explore_num = 0

    def forward(self, data1, data2):
        if not self.use_forward:
            return self.prediction
        # graph_emb [batch_size * 8]
        # sslout [batch_size * 3]
        graph_emb1, graph_emb2, sslout1, sslout2 = self.supernet0(data1, data2, mode="mixed")
        # graph_alpha [num_layers * batch_size * 6], representing 6 options per layer
        graph_alpha1, cosloss1 = self.ag(graph_emb1)
        graph_alpha2, cosloss2 = self.ag(graph_emb2)
        # pred [batch_size * 1]
        # emb [batch_size * 128]
        node_embs1, node_mask1 = self.supernet(data1, mode="mads", graph_alpha=graph_alpha1)
        node_embs2, node_mask2 = self.supernet(data2, mode="mads", graph_alpha=graph_alpha2)
        return cosloss1, sslout1, node_embs1, node_mask1, cosloss2, sslout2, node_embs2, node_mask2

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graces")

        parser.add_argument(
            "--supernet_learning_rate", type=float, default=5e-4, help="init learning rate"
        )
        parser.add_argument(
            "--supernet_learning_rate_min", type=float, default=1e-4, help="min learning rate"
        )
        parser.add_argument("--supernet_weight_decay", type=float, default=5e-4, help="weight decay")
        parser.add_argument(
            "--epsilon",
            type=float,
            default=0.0,
            help="the explore rate in the gradient descent process",
        )
        parser.add_argument(
            "--arch_learning_rate",
            type=float,
            default=0.01,
            help="learning rate for arch encoding",
        )
        parser.add_argument(
            "--arch_learning_rate_min",
            type=float,
            default=0.001,
            help="minimum learning rate for arch encoding",
        )
        parser.add_argument(
            "--arch_weight_decay",
            type=float,
            default=1e-4,
            help="weight decay for arch encoding",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            type=float,
            default=0.05,
            help="learning rate for arch encoding",
        )
        parser.add_argument(
            "--encoder_learning_rate_min",
            type=float,
            default=0.005,
            help="minimum learning rate for arch encoding",
        )
        parser.add_argument(
            "--encoder_weight_decay",
            type=float,
            default=1e-4,
            help="weight decay for arch encoding",
        )
        parser.add_argument(
            "--pooling_ratio", type=float, default=0.5, help="global pooling ratio"
        )
        parser.add_argument("--beta", type=float, default=5e-3, help="cosloss weight")
        parser.add_argument("--gamma", type=float, default=5e-3, help="sslloss weight")
        parser.add_argument("--eta", type=float, default=0.1, help="errorloss weight")
        parser.add_argument(
            "--with_conv_linear",
            type=bool,
            default=False,
            help=" in NAMixOp with linear op",
        )
        parser.add_argument(
            "--num_layers", type=int, default=3, help="num of layers of GNN method."
        )
        parser.add_argument(
            "--search_act",
            action="store_true",
            default=False,
            help="search act in supernet.",
        )
        parser.add_argument(
            "--hidden_size", type=int, default=128, help="default hidden_size in supernet"
        )
        parser.add_argument(
            "--graph_dim", type=int, default=8, help="default hidden_size in supernet"
        )
        parser.add_argument(
            "--dropout", type=float, default=0.5, help="default hidden_size in supernet"
        )
        # in the stage of update theta.
        parser.add_argument(
            "--temp", type=float, default=0.2, help="temperature in gumble softmax."
        )
        parser.add_argument(
            "--loc_mean",
            type=float,
            default=10.0,
            help="initial mean value to generate the location",
        )
        parser.add_argument(
            "--loc_std",
            type=float,
            default=0.01,
            help="initial std to generate the location",
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default="darts",
            help="how to update alpha",
            choices=["mads", "darts", "snas"],
        )
        parser.add_argument(
            "--temperature", type=float, default=1, help="temperature of AGLayer"
        )
        parser.add_argument(
            "--use_att", action='store_true', help="wether to use graph attention"
        )
        parser.add_argument(
            "--att_head", type=int, default=4, help="graph attention heads"
        )
        return parent_parser