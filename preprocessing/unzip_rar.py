import patoolib

def extract_rar(file_path, output_dir):
    """
    Extract a .rar file to the specified output directory.

    :param file_path: Path to the .rar file
    :param output_dir: Directory where the files will be extracted
    """
    try:
        patoolib.extract_archive(file_path, outdir=output_dir)
        print(f"Extraction of {file_path} complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
rar_file_path = "data.rar"
output_directory = "../data"
extract_rar(rar_file_path, output_directory)