import csv
from collections import defaultdict

def process_tracking_file(track_file_path, output_track_file_path):
    # Read the track file and store its contents in track_data list
    track_data = []
    id_license_mapping = defaultdict(str)  # Dictionary to store the mapping of license plates to IDs

    with open(track_file_path, 'r') as track_file:
        csv_reader = csv.reader(track_file)
        header = next(csv_reader)  # Skip the header
        for row in csv_reader:
            track_data.append(row)
            frame_number, id_, _, _, _, _, license_plate, _ = row
            if license_plate != ' None':
                if license_plate in id_license_mapping:
                    # If the license plate is already associated with an ID, update the current ID
                    row[1] = id_license_mapping[license_plate]
                else:
                    # Otherwise, store the current ID for this license plate
                    id_license_mapping[license_plate] = id_

    # Write the modified track data to a new file
    with open(output_track_file_path, 'w', newline='') as output_track_file:
        csv_writer = csv.writer(output_track_file)
        csv_writer.writerow(header)  # Write the header
        csv_writer.writerows(track_data)

    print("New track file created with corrected IDs.")

# Example usage
process_tracking_file(r'Path to the tracking file with the license plates', r'Output file path of the post-process')
