import tqdm
import multiprocessing
import json

def process_row(something):
    print("hest")


if __name__ == "__main__":

    NUM_PROCESSES = 20

    dataset = "load from somewhere"
    rows_to_process = dataset

    if not rows_to_process:
        print("All items processed!")
        exit()

    print(
        f"Starting processing for {len(rows_to_process)} items with {NUM_PROCESSES} processes..."
    )

    with open(output_file, "a", encoding="utf-8") as file:
        with multiprocessing.Pool(
            processes=NUM_PROCESSES, initializer=init_worker
        ) as pool:
            results = pool.imap_unordered(process_row, rows_to_process, chunksize=1)
            for result in tqdm(results, total=len(rows_to_process)):

                if result:
                    file.write(json.dumps(result) + "\n")
                    file.flush()