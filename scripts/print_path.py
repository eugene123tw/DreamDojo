import os


if __name__ == "__main__":
    dataset_root = "/mnt/amlfs-03/shared/datasets/Gear2025CoRLDemo"
    subsets = os.listdir(dataset_root)
    print_path = []
    for subset in subsets:
        subset_path = os.path.join(dataset_root, subset)
        if os.path.isdir(subset_path) and os.listdir(subset_path):
            print_path.append("      - " + subset_path)

    print_path = sorted(print_path)
    print("\n".join(print_path))
