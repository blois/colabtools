"""Helper for constructing interactive tables."""
import cgi
import json
import numbers
import types

import pandas as pd


def IsNumpyType(x):
  """Returns true if the value is a numpy type."""
  return 'numpy' in str(type(x))


def FindFormatter(formatters):
  """Returns a formatter that takes x, and applies formatter based on types.

  Args:
    formatters: map from type to formatter

  Returns:
    function: x -> displayable output
  """

  def Formatter(x):
    for each, formatter in formatters.items():
      if isinstance(x, each):
        return formatter(x)
    return x

  return Formatter


_np_int_types = ('int32', 'int64', 'int8', 'int16', 'uint32', 'uint64', 'uint8',
                 'uint16')


def ProcessCustomFormatters(formatters, columns):
  """Re-keys a dict of custom formatters to only use column indices.

  Args:
    formatters: A dict of formatters, keyed by column index or name.
    columns: The list of columns names.

  Returns:
    A dict of formatters keyed only by column index.
  """
  if not formatters:
    return {}

  # Check that all keys provided are valid column names or indices.
  # Warn if something doesn't check out.
  column_set = set(columns)
  for col in formatters:
    if isinstance(col, int) and col >= len(columns):
      print('Warning: Custom formatter column index %d exceeds total number '
            'of columns (%d)') % (col, len(columns))

    if not isinstance(col, int) and col not in column_set:
      print('Warning: Custom formatter column name %s not present in column '
            'list') % col

  # Separate out the custom formatters that use indices.
  output_formatters = {
      k: v
      for k, v in formatters.items() if isinstance(k, int)
  }

  for i, name in enumerate(columns):
    # Attempt to find a formatter based on column name.
    if name in formatters:
      if i in output_formatters:
        print('Warning: Custom formatter for column index %d present, '
              'ignoring formatter for column name %s') % (i, name)
      else:
        output_formatters[i] = formatters[name]

  return output_formatters


def ToJs(x, default_nonunicode_formatter, formatter=None, as_string=False):
  """Formats given x into js-parseable structure.

  Args:
    x: describes the data
    default_nonunicode_formatter: The default formatter to use for non-Unicode.
    formatter: function-like object that takes x and returns html string.
    as_string: force the value to be a string in the JSON.
  Returns:
    string - the javascript representation
  """
  if formatter is not None:
    x = formatter(x)
  # Making the output a list, causes datatable interpret its elements
  # as html, rather than text.
  if hasattr(x, '__html__'):
    x = x.__html__()
  elif hasattr(x, '_repr_html_'):
    x = x._repr_html_()  # pylint: disable=protected-access

  # These converters are meant to produce reasonable values
  # but for anything customizables users should just create per-type
  # converters in interactive_table.DEFAULT_FORMATTERS
  if IsNumpyType(x) and hasattr(x, 'dtype'):
    if x.dtype.kind == 'M':
      x = str(pd.to_datetime(x))
    elif x.shape:
      # Convert lists into their string representations
      x = str(x)
    elif x.dtype.kind == 'b':
      x = bool(x)
    elif type(x).__name__.startswith('float'):
      if hasattr(x, 'is_integer') and x.is_integer():
        x = int(x)
      else:
        x = float(x)
    elif type(x).__name__ in _np_int_types:
      x = long(x)
  if isinstance(x, (int, long)) and abs(x) > 2**52:
    x = '%d' % x

  # Ensure that we're returning JSON of a string value.
  double_encode_json = as_string and not isinstance(x, basestring)

  try:
    result = json.dumps(x, default=lambda x: cgi.escape(str(x)))
  except UnicodeDecodeError:
    if isinstance(x, basestring):
      result = json.dumps(default_nonunicode_formatter(x))
    else:
      result = json.dumps([ToJs(el, default_nonunicode_formatter) for el in x])
  result = result.replace('</', '<\\/')
  if double_encode_json:
    result = json.dumps(result)
  return result


def ToJsMatrix(matrix, default_nonunicode_formatter, custom_formatters,
               max_data_size):
  """Creates a two dimensional javascript compatible matrix.

  Args:
      matrix: is any iterator-of-iterator matrix. Currently
        the individual type should be numbers of strings.
      default_nonunicode_formatter: The default formatter to use for
         non-Unicode.
      custom_formatters: a map that provides custom formatters
         for some or all columns.
      max_data_size: maximum size allowed, if exceeds, the remaining rows will
      be dropped

  Returns:
     javascript representation.
  """
  values = ((ToJs(el, default_nonunicode_formatter,
                  custom_formatters.get(i, None)) for i, el in enumerate(row))
            for row in matrix)
  values = [','.join((value for value in row)) for row in values]
  total = 0
  discarded = 0
  i = len(values)
  for i, each in enumerate(values):
    total += len(each) + 10
    if total > max_data_size:
      discarded = len(values) - i
      values = values[:i]
      break
  if discarded:
    print('The table data exceeds the limit %d. Will discard last %d rows ' %
          (max_data_size, discarded))
  return '[[%s]]' % ('],\n ['.join(values))


def TrimColumns(columns, max_columns):
  """Prints a warning and returns trimmed columns if necessary."""
  if len(columns) <= max_columns:
    return columns
  print('Warning: Total number of columns (%d) exceeds max_columns (%d)'
        ' limiting to first max_columns ') % (len(columns), max_columns)
  return columns[:max_columns]


def TrimData(data, max_rows, max_columns=None):
  """Prints a warning and returns trimmed data if necessary."""

  # If the number of columns per row exceeds the max, we need to trim each row.
  if max_columns is not None and len(data) and len(data[0]) > max_columns:
    for i, _ in enumerate(data):
      data[i] = data[i][:max_columns]

  if len(data) <= max_rows:
    return data
  print('Warning: total number of rows (%d) exceeds max_rows (%d) '
        ' limiting to first max_rows ') % (len(data), max_rows)
  return data[:max_rows]


_NUMBER_TYPES = ('int', 'uint', 'long', 'float')
_ALLOWED_TYPES = _NUMBER_TYPES + ('string', 'NoneType')


def DetermineColumnType(data_types):
  """Given a set of Python column types, returns either 'number' or 'string'."""
  # Allow None which will be converted to NaN.
  if all(
      issubclass(t, (numbers.Number, types.NoneType)) and
      not issubclass(t, bool) for t in data_types):
    return 'number'
  return 'string'


def GetColumnType(data, column_index):
  """Returns the best-guess JS type for the column in the data."""
  data_types = set()
  for row in data:
    cell = row[column_index]
    t = type(cell)
    is_known_type = (cell is None or issubclass(t, numbers.Number) or
                     issubclass(t, basestring))
    if not is_known_type:
      t = str
    data_types.add(t)
  return DetermineColumnType(data_types)


def FormatData(data, default_formatter, custom_formatters):
  """Formats the given data and determines column types."""
  num_columns = len(data[0])
  column_types = [GetColumnType(data, i) for i in range(num_columns)]
  formatted_values = []
  for row in data:
    formatted_row = []
    for column_index, cell in enumerate(row):
      custom_formatter = custom_formatters.get(column_index, None)
      formatted_value = custom_formatter(cell) if custom_formatter else cell
      column_type = column_types[column_index]
      if column_type is not 'number' or not custom_formatter:
        formatted_row.append(ToJs(formatted_value, default_formatter))
      else:
        raw_value = ToJs(cell, default_formatter)
        formatted_value = ToJs(
            formatted_value, default_formatter, as_string=True)
        formatted_row.append("""{
            'v': %s,
            'f': %s,
        }""" % (raw_value, formatted_value))

    formatted_values.append(',\n'.join(formatted_row))

  formatted_data = '[[%s]]' % ('],\n ['.join(formatted_values))

  return {'column_types': column_types, 'data': formatted_data}
