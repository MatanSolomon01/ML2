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
    def basic_transform(df, **kwargs):
        df = df.drop(df.columns[kwargs['columns_index_to_drop']], axis=1) if 'columns_index_to_drop' in kwargs else df
        df = df.dropna().reset_index(drop=True) if ('remove_null_rows' in kwargs and kwargs['remove_null_rows']) else df

        return df

    @staticmethod
    def print_constants():
        print("Run Constants")
        with open('Constants.py', 'r') as constants:
            for line in constants:
                print('\t', line, end='')
        print('\n')
