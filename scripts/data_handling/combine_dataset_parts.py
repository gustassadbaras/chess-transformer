from pathlib import Path
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

def combine_parquet_files(input_dir: Path, output_file: Path) -> None:
    files = sorted(input_dir.glob("part_*"))
    files = [f for f in files if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No files matching 'part_*' in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = [f for f in files if f.resolve() != output_file.resolve()]

    if not files:
        raise ValueError("No input files remain after excluding the output file.")

    writer = None
    try:
        for file in files:
            parquet_file = pq.ParquetFile(file)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_file),
                    parquet_file.schema_arrow,
                    compression="snappy",
                )

            for batch in parquet_file.iter_batches(batch_size=65_536):
                table = pa.Table.from_batches(
                    [batch],
                    schema=parquet_file.schema_arrow,
                )
                writer.write_table(table, row_group_size=65_536)

            parquet_file.close()
    finally:
        if writer is not None:
            writer.close()

    print(f"Combined {len(files)} files into: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    combine_parquet_files(args.input_dir, args.output_file)