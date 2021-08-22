from sklearn.preprocessing import MinMaxScaler


class Utils:

    @staticmethod
    def split(string, delimiters):
        """
            Split strings according to delimiters
            :param string: full sentence
            :param delimiters string: characters for spliting
                function splits sentence to words
        """
        delimiters = tuple(delimiters)
        stack = [string, ]

        for delimiter in delimiters:
            for i, substring in enumerate(stack):
                substack = substring.split(delimiter)
                stack.pop(i)
                for j, _substring in enumerate(substack):
                    stack.insert(i + j, _substring)

        return stack

    @staticmethod
    def basic_transform(df, basic_transform=None):
        if basic_transform is None:
            return df
        for transformation, args in basic_transform.items():
            df = transformation(df, *args)
        return df

    @staticmethod
    def print_constants():
        print("Run Constants")
        with open('Constants.py', 'r') as constants:
            for line in constants:
                print('\t', line, end='')
        print('\n')


class DfOperations:
    @staticmethod
    def drop_columns(df, columns_index_to_drop):
        return df.drop(df.columns[columns_index_to_drop], axis=1)

    @staticmethod
    def drop_null_rows(df):
        return df.dropna().reset_index(drop=True)

    @staticmethod
    def normalize(df, method):
        if method == 'MinMax':
            scaler = MinMaxScaler()
            df[:] = scaler.fit_transform(df)
        return df

    @staticmethod
    def omit_last(df, last):
        return df.iloc[:-last, :]

    @staticmethod
    def leave_last(df, last):
        return df.iloc[-last:, :]
