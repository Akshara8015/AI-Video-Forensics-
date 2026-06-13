print('Training...')

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_closs = None
    running_sloss = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))

        running_loss = 0
        running_closs = 0
        running_sloss = 0

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            
            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step()

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)






# Add F1 Score
# from sklearn.metrics import f1_score
# all_preds = []
# all_labels = []

# all_preds.extend(
#     preds.cpu().numpy()
# )

# all_labels.extend(
#     labels.cpu().numpy()
# )


# f1 = f1_score(
#     all_labels,
#     all_preds
# )

# return epoch_loss, epoch_acc, f1

# print(
#     f"Val F1 Score: {f1:.4f}"
# )


# Accuracy
# Precision
# Recall
# F1 Score
# ROC-AUC
# Confusion Matrix

