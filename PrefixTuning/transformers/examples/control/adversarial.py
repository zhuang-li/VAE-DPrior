import torch
from torch import nn

from autoencoders.mapping import ResNet, CosineLoss



class Adversarial(nn.Module):
    def __init__(self, critic_hidden_units, embedding_dim, critic_hidden_layers, critic_lr, device):
        super(Adversarial, self).__init__()
        hidden = critic_hidden_units
        critic_layers = [nn.Linear(embedding_dim,
                                   hidden),
                         nn.ReLU()]
        for _ in range(critic_hidden_layers):
            critic_layers.append(nn.Linear(hidden, hidden))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(hidden, 2))
        self.critic = [nn.Sequential(*critic_layers)]

        self.generator = nn.Linear(embedding_dim,
                                   hidden).to(device)

        self.critic_loss = nn.CrossEntropyLoss()
        self.critic_optimizer = torch.optim.Adam(
            self._get_critic().parameters(), lr=critic_lr)
        dev = device
        self._get_critic().to(dev)
        self.critic_loss.to(dev)



    def forward(self, style_embeddings, topic_embeddings):

        # train the discriminator
        # NOTE: We need to train the discriminator first, because otherwise we would
        # backpropagate through the critic after changing it

        self._train_critic(style_embeddings, topic_embeddings)

        critic_loss = self._test_critic(topic_embeddings)

        # train critic
        self.gen_optimizer.zero_grad()
        critic_loss.backward()
        self.gen_optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)


        return critic_loss.item()


    #def forward(self, rand_emb):
    #    return self.generator(rand_emb)


    def _test_critic(self, embeddings):
        self._get_critic().eval()

        # with torch.no_grad():
        # do not detach embeddings, because we need to propagate through critic
        # within the same computation graph
        critic_logits = self._get_critic()(embeddings)

        labels = torch.ones(
            (embeddings.shape[0]), device=embeddings.device, dtype=torch.long)
        #print(labels.size())
        loss = self.critic_loss(critic_logits, labels)
        return loss

    def _get_critic(self):
        return self.critic[0]

    def _train_critic(self, real_embeddings, generated_embeddings):
        self._get_critic().train()

        # need to detach from the current computation graph, because critic has
        # its own computation graph
        real_embeddings = real_embeddings.detach().clone()
        generated_embeddings = generated_embeddings.detach().clone()

        # get predictions from critic
        all_embeddings = torch.cat(
            [real_embeddings, generated_embeddings], dim=0)
        critic_logits = self._get_critic()(all_embeddings)

        # compute critic loss
        true_labels = torch.ones(
            (real_embeddings.shape[0]), device=real_embeddings.device, dtype=torch.long)
        false_labels = torch.zeros(
            (generated_embeddings.shape[0]), device=generated_embeddings.device, dtype=torch.long)
        labels = torch.cat([true_labels, false_labels], dim=0)
        loss = self.critic_loss(critic_logits, labels)

        # train critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return loss