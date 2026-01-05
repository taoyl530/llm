def get_main_arguments(parser):
    """Required parameters"""

    parser.add_argument("--model_name", 
                        default='sasrec_seq',
                        choices=['llm4cdsr'],
                        type=str, 
                        required=False,
                        help="model name")
    parser.add_argument("--dataset", 
                        default="douban", 
                        choices=["douban", "amazon", "elec","movies", # preprocess by myself
                                ], 
                        help="Choose the dataset")
    parser.add_argument("--domain",
                        default="0",
                        type=str,
                        help="the domain flag for SDSR")
    parser.add_argument("--inter_file",
                        default="cloth_sport",
                        type=str,
                        help="the name of interaction file")
    parser.add_argument("--pretrain_dir",
                        type=str,
                        default="sasrec_seq",
                        help="the path that pretrained model saved in")
    parser.add_argument("--output_dir",
                        default='./saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--check_path",
                        default='',
                        type=str,
                        help="the save path of checkpoints for different running")
    parser.add_argument("--do_test",
                        default=False,
                        action="store_true",
                        help="whehther run the test on the well-trained model")
    parser.add_argument("--do_emb",
                        default=False,
                        action="store_true",
                        help="save the user embedding derived from the SRS model")
    parser.add_argument("--do_group",
                        default=False,
                        action="store_true",
                        help="conduct the group test")
    parser.add_argument("--do_cold",
                        default=False,
                        action="store_true",
                        help="whether test cold start")
    parser.add_argument("--ts_user",
                        type=int,
                        default=10,
                        help="the threshold to split the short and long seq")
    parser.add_argument("--ts_item",
                        type=int,
                        default=20,
                        help="the threshold to split the long-tail and popular items")
    # --- New: multi-GPU options ---
    parser.add_argument("--use_dp",
                        action='store_true',
                        help="Enable DataParallel to split batches across multiple GPUs")
    parser.add_argument("--dp_gpu_ids",
                        type=str,
                        default='',
                        help="Comma-separated GPU ids for DataParallel, e.g. '0,1'. Empty uses all available")
    parser.add_argument("--llm_item_device",
                        type=str,
                        default='',
                        help="Device for global item LLM embedding, e.g. 'cuda:1'. Empty keeps default")
    return parser


def get_model_arguments(parser):
    """Model parameters"""
    
    parser.add_argument("--hidden_size",
                        default=64,
                        type=int,
                        help="the hidden size of embedding")
    parser.add_argument("--trm_num",
                        default=2,
                        type=int,
                        help="the number of transformer layer")
    parser.add_argument("--num_heads",
                        default=1,
                        type=int,
                        help="the number of heads in Trm layer")
    parser.add_argument("--num_layers",
                        default=1,
                        type=int,
                        help="the number of GRU layers")
    parser.add_argument("--cl_scale",
                        type=float,
                        default=0.1,
                        help="the scale for contastive loss")
    parser.add_argument("--tau",
                        default=1,
                        type=float,
                        help="the temperature for contrastive loss")
    parser.add_argument("--tau_reg",
                        default=1,
                        type=float,
                        help="the temperature for contrastive loss")
    parser.add_argument("--dropout_rate",
                        default=0.5,
                        type=float,
                        help="the dropout rate")
    parser.add_argument("--max_len",
                        default=200,
                        type=int,
                        help="the max length of input sequence")
    parser.add_argument("--mask_prob",
                        type=float,
                        default=0.6,
                        help="the mask probability for training Bert model")
    parser.add_argument("--mask_crop_ratio",
                        type=float,
                        default=0.3,
                        help="the mask/crop ratio for CL4SRec")
    parser.add_argument("--aug",
                        default=False,
                        action="store_true",
                        help="whether augment the sequence data")
    parser.add_argument("--aug_seq",
                        default=False,
                        action="store_true",
                        help="whether use the augmented data")
    parser.add_argument("--aug_seq_len",
                        default=0,
                        type=int,
                        help="the augmented length for each sequence")
    parser.add_argument("--aug_file",
                        default="inter",
                        type=str,
                        help="the augmentation file name")
    parser.add_argument("--train_neg",
                        default=1,
                        type=int,
                        help="the number of negative samples for training")
    parser.add_argument("--test_neg",
                        default=100,
                        type=int,
                        help="the number of negative samples for test")
    parser.add_argument("--suffix_num",
                        default=5,
                        type=int,
                        help="the suffix number for augmented sequence")
    parser.add_argument("--prompt_num",
                        default=2,
                        type=int,
                        help="the number of prompts")
    parser.add_argument("--freeze",
                        default=False,
                        action="store_true",
                        help="whether freeze the pretrained architecture when finetuning")
    parser.add_argument("--freeze_emb",
                        default=False,
                        action="store_true",
                        help="whether freeze the embedding layer, mainly for LLM embedding")
    parser.add_argument("--alpha",
                        default=0.1,
                        type=float,
                        help="the weight of auxiliary loss")
    parser.add_argument("--beta",
                        default=0.1,
                        type=float,
                        help="the weight of regulation loss")
    parser.add_argument("--llm_emb_file",
                        default="item_emb",
                        type=str,
                        help="the file name of the LLM embedding")
    parser.add_argument("--expert_num",
                        default=1,
                        type=int,
                        help="the number of adapter expert")
    parser.add_argument("--user_emb_file",
                        default="user_emb",
                        type=str,
                        help="the file name of the user LLM embedding")
    # for LightGCN
    parser.add_argument("--layer_num",
                        default=2,
                        type=int,
                        help="the number of collaborative filtering layers")
    parser.add_argument("--keep_rate",
                        default=0.8,
                        type=float,
                        help="the rate for dropout")
    parser.add_argument("--reg_weight",
                        default=1e-6,
                        type=float,
                        help="the scale for regulation of parameters")
    # --- New: Cross-Domain GNN switches ---
    parser.add_argument("--use_gnn",
                        default=False,
                        action="store_true",
                        help="whether use GNN for structure enhancement")
    parser.add_argument("--gnn_weight",
                        default=0.5,
                        type=float,
                        help="fusion weight for GNN structure embedding with semantic embedding")
    # for LLM4CDSR
    parser.add_argument("--local_emb",
                        default=False,
                        action="store_true",
                        help="whether use the LLM embedding to initilize the local embedding")
    parser.add_argument("--global_emb",
                        default=False,
                        action="store_true",
                        help="whether use the LLM embedding to substitute global embedding")
    parser.add_argument("--thresholdA",
                        default=0.5,
                        type=float,
                        help="mask rate for AMID")
    parser.add_argument("--thresholdB",
                        default=0.5,
                        type=float,
                        help="mask rate for AMID")
    parser.add_argument("--hidden_size_attr",
                        default=32,
                        type=int,
                        help="the hidden size of attribute embedding")

    # === [LLM-Guided Graph Structure Learning 参数] ===
    parser.add_argument('--graph_prune_threshold', type=float, default=0.1,
                        help='Threshold of cosine similarity for pruning noisy edges in LLM-guided graph refinement.')
    parser.add_argument('--graph_aug_enable', type=int, default=1,
                        help='Whether to enable cold-start augmentation (1 for True, 0 for False).')
    parser.add_argument('--graph_cold_threshold', type=int, default=5,
                        help='Degree threshold to identify cold-start users for augmentation.')
    parser.add_argument('--graph_aug_k', type=int, default=2,
                        help='Number of semantic edges to add for each cold-start user as candidates.')   
    
    # [AGSL New Arguments]
    parser.add_argument('--graph_learn_tau', type=float, default=1.0,
                        help='Temperature for Gumbel-Softmax in graph structure learning.')
    parser.add_argument('--graph_learn_weight', type=float, default=0.01,
                        help='Weight for the graph learning distillation loss (alignment with LLM priors).')
    
    # === [NEW] Semantic Alignment 参数 ===
    parser.add_argument('--align_weight', type=float, default=0.1,
                        help='Weight for the Semantic-Behavior Alignment (InfoNCE) loss.')
    parser.add_argument('--align_tau', type=float, default=0.2,
                        help='Temperature for the Alignment InfoNCE loss.')
    
    return parser


def get_train_arguments(parser):

    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform")
    parser.add_argument("--lr",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for AdamW optimizer")
    parser.add_argument("--l2",
                        default=0.0,
                        type=float,
                        help="Weight decay (L2 regularization) for optimizer")
    parser.add_argument("--lr_dc_step",
                        default=10,
                        type=int,
                        help="Step size for learning rate decay scheduler")
    parser.add_argument("--lr_dc",
                        default=0.1,
                        type=float,
                        help="Gamma (decay rate) for learning rate scheduler")
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="Batch size for training")
    parser.add_argument("--valid_batch_size",
                        default=64,
                        type=int,
                        help="Batch size for validation")
    parser.add_argument("--test_batch_size",
                        default=64,
                        type=int,
                        help="Batch size for testing")
    parser.add_argument("--patience",
                        type=int,
                        default=10,
                        help="the patience number for early stop")
    parser.add_argument("--watch_metric",
                        type=str,
                        default='NDCG@10',
                        choices=['NDCG@10', 'HR@10', 'MRR@10', 'MAP@10'],
                        help="Which metric key to watch for early stop and model selection")
    parser.add_argument("--topk",
                        type=int,
                        default=10,
                        help="the top-k items for other metric")
    parser.add_argument("--metrics",
                        type=str,
                        default='ndcg',
                        choices=['recall', 'ndcg', 'mrr', 'map'],
                        help="which metric is used to select model.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for different data split")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gpu_id',
                        default=0,
                        type=int,
                        help='The device id.')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='The number of workers in dataloader')
    parser.add_argument("--log", 
                        default=False,
                        action="store_true",
                        help="whether create a new log file")
    
    return parser