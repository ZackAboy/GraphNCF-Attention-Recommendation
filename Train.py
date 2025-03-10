import torch
from Evaluation import Evaluation

class Train:
    def __init__(self,
                 train_loader,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion,
                 optim: torch.optim,
                 epochs: int,
                 test_loader,
                 top_k: int,
                 edge_index=None,
                 edge_weight=None):
        self.epochs = epochs
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optim
        self.dataloader = train_loader
        self.test_loader = test_loader
        self.top_k = top_k
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def train(self):
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        dataloader = self.dataloader
        device = self.device
        top_k = self.top_k

        for epoch in range(self.epochs):
            avg_cost = 0
            total_batch = len(dataloader)

            for idx, (users, pos_items, neg_items) in enumerate(dataloader):
                users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)

                user_embeddings, pos_item_embeddings, neg_item_embeddings = model(
                    users, pos_items, neg_items,
                    edge_index=self.edge_index,
                    edge_weight=self.edge_weight,
                    use_dropout=True
                )

                optimizer.zero_grad()
                cost = criterion(user_embeddings, pos_item_embeddings, neg_item_embeddings)
                cost.backward()
                optimizer.step()

                avg_cost += cost.item()

            avg_cost /= total_batch

            eval = Evaluation(test_dataloader=self.test_loader,
                              model=model,
                              top_k=top_k,
                              device=device)
            HR, NDCG = eval.get_metric(edge_index=self.edge_index, edge_weight=self.edge_weight)
            print(f'Epoch: {(epoch + 1):04}, {criterion._get_name()}= {avg_cost:.9f}, NDCG@{top_k}:{NDCG:.4f}, HR@{top_k}:{HR:.4f}')

        print("Training Completed.")