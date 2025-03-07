import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class block(nn.Module):

    def __init__(self, in_chanells, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_chanells, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):

    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        self.upsample1 = nn.ConvTranspose2d(512 * 4, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample5 = nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        x = x.unsqueeze(1)
        return x
    
    def _make_layer(self, block, num_residual, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        for i in range(num_residual-1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


class TuSimpleLaneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(os.listdir(os.path.join(root_dir, "frames")))
        self.mask_paths = sorted(os.listdir(os.path.join(root_dir, "lane-masks")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "frames", self.image_paths[idx])
        mask_path = os.path.join(self.root_dir, "lane-masks", self.mask_paths[idx])

        # Load and process the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (224, 224))  # Resize for consistency

        # Load and process the mask (binary: lane=1, background=0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        mask = (mask > 128).astype(np.uint8)  # Convert to binary (0 or 1)

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
        mask = torch.tensor(mask, dtype=torch.long)  # Keep as class index

        return image, mask

def ResNet152(img_channels=3, num_classes=1):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)

model = ResNet152()
model = model.to("cuda")
model.load_state_dict(torch.load("resnet152_V3.pth"))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 1000

def train(model, data, lossfn, optimizer, epochs):
    
    model.train()
    for epoch in range(epochs):
        for images, masks in data:
            images, masks = images.to("cuda"), masks.to("cuda").float()
            masks = masks.unsqueeze(1)
            # print(f"Mask Shape: {masks.shape}")
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = lossfn(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), r"C:\Users\lambo\Desktop\archive\TUSimple\resnet152_V3.pth")

def predict(model, image_path):

    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to("cpu")
    with torch.inference_mode():
        output = model(image_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (mask > 0.5).astype(np.uint8) * 255  
    return mask

def process_frame(frame):
    """Processes a single video frame through the lane segmentation model and overlays the entire lane."""
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to('cuda')

    
    with torch.inference_mode():
        output = model(image_tensor)
        mask = torch.sigmoid(output).cpu().numpy().squeeze()  # Remove extra dimensions
        mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask (0 or 255)s

    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size if needed
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)  # Closes holes in the mask
 
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_lane = np.zeros_like(mask_resized, dtype=np.uint8)
    

    cv2.drawContours(filled_lane, contours, -1, 255, thickness=cv2.FILLED)  # Fill the lane

    filled_lane_colored = np.stack([filled_lane] * 3, axis=-1)  # Convert (H, W) → (H, W, 3)
    lane_overlay = np.zeros_like(frame, dtype=np.uint8)
    lane_overlay[:] = (0, 255, 0)  # Green overlay for the entire lane


    lane_overlay = cv2.bitwise_and(lane_overlay, lane_overlay, mask=filled_lane)
    alpha = 0.5  # Transparency level
    frame = cv2.addWeighted(frame, 1 - alpha, lane_overlay, alpha, 0)

    return frame


def process_video(input_video_path, output_video_path):
    """Reads a video, applies lane segmentation, and saves the output."""
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Process the frame through the model
        processed_frame = process_frame(frame)
        # cv2.imwrite(r"C:\Users\lambo\Desktop\archive\TUSimple\single_frame.jpg", frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Show frame (optional for debugging)
        cv2.imshow("Lane Segmentation", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to stop early
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete! Saved to:", output_video_path)

# Example Usage:



dataset = TuSimpleLaneDataset(root_dir=r"C:\Users\lambo\Desktop\archive\TUSimple")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#train(model, data=dataloader, lossfn=loss_fn, optimizer=optimizer, epochs=epochs)
#Example Prsediction
pred_mask = predict(model, r"C:\Users\lambo\Desktop\archive\TUSimple\test_set\clips\0531\1492626272918083058\20.jpg")
print(f"Predicted Mask Shape: {pred_mask.shape}")
pred_mask = pred_mask.squeeze(0)
#cv2.imwrite(r"C:\Users\lambo\Desktop\archive\TUSimple\predicted_mask.jpg", pred_mask)
process_video("input_video.mp4", "output_video.mp4")
