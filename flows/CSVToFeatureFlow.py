from metaflow import FlowSpec, step, pypi
import csv
import pandas as pd

class CsvToFeatureFlow(FlowSpec):

    @pypi(python='3.12.3', packages={'pandas': '2.2.3'})

    @step
    def start(self):
        import pandas as pd
        # Path to your CSV file
        self.csv_file = "experiments-multi/experiments_geo.csv"
        # Read the CSV file
        self.df = pd.read_csv(self.csv_file)
        # Initialize a list to hold features and a dict for fire_periods if needed
        self.features = []
        self.fire_periods = {}
        self.next(self.read_csv)

    @step
    def read_csv(self):
        with open(self.df, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract spatial extent
                SPATIAL_EXTENT = {
                    "west": float(row["west"]),
                    "south": float(row["south"]),
                    "east": float(row["east"]),
                    "north": float(row["north"])
                }

                # Convert to Polygon coordinates
                coordinates = [[
                    [SPATIAL_EXTENT["west"], SPATIAL_EXTENT["south"]],
                    [SPATIAL_EXTENT["east"], SPATIAL_EXTENT["south"]],
                    [SPATIAL_EXTENT["east"], SPATIAL_EXTENT["north"]],
                    [SPATIAL_EXTENT["west"], SPATIAL_EXTENT["north"]],
                    [SPATIAL_EXTENT["west"], SPATIAL_EXTENT["south"]]
                ]]

                # Extract temporal extent
                TEMPORAL_EXTENT = [row["fire_start"], row["fire_end"]]

                # Construct the feature
                feature = {
                    "type": "Feature",
                    "properties": {
                        "temporal_extent": TEMPORAL_EXTENT
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
                    }
                }

                self.features.append(feature)

        self.next(self.end)

    @step
    def end(self):
        # Output the created features
        print("Pipeline finished. Created features:")
        for feature in self.features:
            print(feature)

if __name__ == '__main__':
    CsvToFeatureFlow()
