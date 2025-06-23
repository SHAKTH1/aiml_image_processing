import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from data_utils import unzip_images, load_csvs, split_data, export_to_folders
import os

st.set_page_config(page_title="SmartLabel", layout="wide")


def count_images(root: str) -> int:
    """Recursively count .jpg/.jpeg/.png files under `root`."""
    return sum(
        1
        for _dirpath, _dirs, files in os.walk(root)
        for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: uploads & parameters
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Configuration")
csv_files = st.sidebar.file_uploader(
    "Upload CSV(s)", type='csv', accept_multiple_files=True
)
zip_file = st.sidebar.file_uploader("Upload Images ZIP", type='zip')
test_size = st.sidebar.slider("Test Set Fraction", 0.1, 0.5, 0.2)
sample_n = st.sidebar.slider("Preview Sample Images", 1, 20, 5)


if csv_files and zip_file:
    # ─────────────────────────────────────────────────────────────────────────
    # 0) Save uploads to disk so data_utils can read file paths
    # ─────────────────────────────────────────────────────────────────────────
    UPLOAD_DIR = 'data/uploads'
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 0a) Save the ZIP
    zip_path = os.path.join(UPLOAD_DIR, zip_file.name)
    with open(zip_path, 'wb') as f:
        f.write(zip_file.getbuffer())

    # 1) Unzip images
    images_dir = unzip_images(zip_path, extract_to='data/images')

    # 0b) Save all CSVs, collect their paths
    csv_paths = []
    for up in csv_files:
        p = os.path.join(UPLOAD_DIR, up.name)
        with open(p, 'wb') as f:
            f.write(up.getbuffer())
        csv_paths.append(p)

    # 2) Load & concat CSVs (auto-detect delimiter & filename column)
    try:
        df = load_csvs(csv_paths)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # 3) Compute and display overview metrics
    # ─────────────────────────────────────────────────────────────────────────
    total_images   = len(df)
    labeled_images = count_images(images_dir)

    if 'status' in df.columns:
        ok = int((df.status.str.upper() == 'OK').sum())
        nok = int((df.status.str.upper() != 'OK').sum())
    else:
        ok = labeled_images
        nok = total_images - labeled_images

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Images", total_images)
    c2.metric("Labeled Images", labeled_images)
    c3.metric("OK", ok)
    c4.metric("NOK_count", nok)

    # ─────────────────────────────────────────────────────────────────────────
    # 4) OK vs NOK Distribution (pie chart)
    # ─────────────────────────────────────────────────────────────────────────
    fig = go.Figure(data=[go.Pie(
        labels=["OK", "NOK"],
        values=[ok, nok],
        hole=0.3,
        textinfo='label+percent',
        marker=dict(colors=['#00cc96','#EF553B'])
    )])
    fig.update_layout(
        title="OK vs NOK Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 5) Image Categories
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Image Categories")
    if 'category' in df.columns:
        for cat, cnt in df['category'].value_counts().items():
            st.markdown(f"- {cat}: {cnt}")
    else:
        st.write("_No `category` column found in your CSV(s)._")

    # ─────────────────────────────────────────────────────────────────────────
    # 6) Preview random samples (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📷 Sample Images")
    sample = df.sample(min(sample_n, len(df)))
    cols = st.columns(len(sample))
    for i, (_, row) in enumerate(sample.iterrows()):
        base = row['filename']
        img_path = None
        for ext in ("", ".jpg", ".jpeg", ".png"):
            candidate = base + ext
            for dp, _, files in os.walk(images_dir):
                if candidate in files:
                    img_path = os.path.join(dp, candidate)
                    break
            if img_path:
                break
        if img_path:
            cols[i].image(img_path, caption=row.get('category',''), use_column_width=True)
        else:
            cols[i].warning(f"Missing: {base}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7) Split & Export Data (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    if st.button("➡️ Split & Export Data"):
        train_df, test_df = split_data(df, test_size=test_size)
        export_to_folders(train_df, images_dir, output_dir='output', split_name='train')
        export_to_folders(test_df, images_dir, output_dir='output', split_name='test')
        st.success("Data exported to ./output/train and ./output/test")

    # ─────────────────────────────────────────────────────────────────────────
    # 8) Labeling Progress (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    if 'label_progress' not in st.session_state:
        st.session_state.label_progress = {"Gopal":90, "Bhumika":90, "Nandhini":50}

    st.subheader("Labeling Progress")
    for name, prog in st.session_state.label_progress.items():
        st.write(f"**{name}**")
        st.progress(prog / 100)

    new_name = st.text_input("Add New Labeler Name", key="new_labeler_name")
    new_prog = st.number_input("Progress %", min_value=0, max_value=100, key="new_labeler_prog")
    if st.button("Add Labeler", key="add_labeler"):
        if not new_name or new_name in st.session_state.label_progress:
            st.warning("Enter a unique name and a valid progress value.")
        else:
            st.session_state.label_progress[new_name] = new_prog
            try:
                st.experimental_rerun()
            except:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # 9) Start Training (PyTorch) WITH LIVE UI
    # ─────────────────────────────────────────────────────────────────────────
    if st.button("🚀 Start Training (PyTorch)"):
        # Import torch & torchvision inside the callback to avoid Streamlit watcher issues
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms, models

        st.subheader("🚧 Training in progress…")
        log_area = st.empty()
        prog_bar = st.progress(0)

        # Data transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ]),
        }

        # Datasets & loaders
        train_ds = datasets.ImageFolder('output/train', transform=data_transforms['train'])
        test_ds  = datasets.ImageFolder('output/test',  transform=data_transforms['test'])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=32, shuffle=False)

        # Model setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_ds)
            prog_bar.progress((epoch+1) / num_epochs)
            log_area.text(f"Epoch {epoch+1}/{num_epochs} — Loss: {epoch_loss:.4f}")

        st.success("🎉 Training complete!")

        # Evaluation on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total * 100
        st.write(f"**Test Accuracy:** {acc:.2f}%")

        # Save model
        torch.save(model.state_dict(), "resnet18_finetuned.pth")
        st.write("Model weights saved to `resnet18_finetuned.pth`")

else:
    st.info("Please upload at least one CSV and your images ZIP to begin.")

