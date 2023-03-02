import pandas as pd
import argparse
import uuid
from utils import createConsumer, prepareDataRow

if __name__ == '__main__':
    # create parser to take true position as arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=int)
    args = parser.parse_args()

    dataset = pd.DataFrame()
    recived_msg_count = 0

    consumer = createConsumer()

    for msg in consumer:
        if (recived_msg_count < int(args.records)):
            row: dict = prepareDataRow(msg)
            if 'NaN' in row.values():
                continue
            else:
                data_row = pd.DataFrame.from_records([row])
                dataset = pd.concat([dataset, data_row])
                recived_msg_count += 1
                print(recived_msg_count, row)
        else:
            break

    id = uuid.uuid4()
    dataset.to_csv(f'./../dataset/{id}.csv', index=False)
