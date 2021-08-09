import torch
from .utils import preview


def train_step(i, data, netG, netD, optimizerG, optimizerD, criterion, c):
    ### Update Discriminator: log(D(x)) + log(1 -D(G(z))
    ## Train with all real batch
    netD.zero_grad()
    # Format batch
    real_cpu = data[0].to(c.DEVICE)
    b_size = real_cpu.size(0)
    print("b size", b_size)
    label = torch.full((b_size,), c.real_label, dtype=torch.float, device=c.DEVICE)
    # Forward pass real batch through Discriminator
    output = netD(real_cpu).view(-1)
    # Calculate Loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backwards pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # generate batch of latent vectors
    noise = torch.randn(b_size, c.N_Z, 1, 1, device=c.DEVICE)
    # Generate fake image batch with Generator
    fake = netG(noise)
    label.fill_(c.fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed)
    # with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    # Update Discriminator
    optimizerD.step()

    ### Update Generator: maximize log(D(G(z)))
    netG.zero_grad()
    label.fill_(c.real_label)  # fake labels are real for generator cost
    # Since Discriminator has just been updated, perform another
    # forward pass of all-fake batch through D
    output = netD(fake).view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update Generator
    optimizerG.step()

    return errD, errG, D_x, D_G_z1, D_G_z2


def print_stats(e, E, i, I, errD, errG, D_x, D_G_z1, D_G_z2):
    st = "[{}/{}][{}/{}]\tLoss D: {:.4f}\tLoss G: {:.4f}\t"
    st += "D(x): {:.4f}\tD(G(z)): {:.4f} / {:.4f}"
    st = st.format(e, E, i, I, errD, errG, D_x, D_G_z1, D_G_z2)
    print(st)


def train(dataloader, netG, netD, optimizerG, optimizerD, criterion, c):
    print("TRAINING...")
    for e in range(c.N_EPOCHS):
        for i, data in enumerate(dataloader, 0):
            errD, errG, D_x, D_G_z1, D_G_z2 = train_step(
                i, data, netG, netD, optimizerG, optimizerD, criterion, c
            )

            if i % 1 == 0:
                print_stats(
                    e, c.N_EPOCHS, i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2
                )

            if i % 10 == 0 and i != 0:
                noise = torch.randn(c.BATCH_SIZE, c.N_Z, 1, 1, device=c.DEVICE)
                fake = netG(noise).detach().cpu()
                preview(fake)
        break
