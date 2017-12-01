"""Widget exposing any data as an InteractiveTable in colab, esp. from pandas.

example usage:
cell[1]: browser_columns = ["engine", "browser", "platform", "version"];
data= ([[ "Trident", "Internet Explorer 4.0", "Win 95+", 5, "X" ],
        [ "Trident", "Internet Explorer 5.0", "Win 95+", 555, "C" ]]);

cell[2]: table = interactive_table.InteractiveTable(browser_columns, data * 1);
         table # to actually display the contents
"""
import cgi
import json
import traceback
import uuid

from IPython import display

from google.colab import _interactive_table_helper

# Cells that have nonunicode symbols will be formatted using this formatter.
# If this formatter needs to produce html or other rich output,
# make sure it returns list [html], rather than html, otherwise it will be
# interpreted as text.

FORCE_TO_LATIN1 = lambda x: (  # pylint: disable=g-long-lambda
    'nonunicode data: %s...' % cgi.escape(x[:100].decode('latin1')))

DEFAULT_NONUNICODE_FORMATTER = FORCE_TO_LATIN1

DEFAULT_FORMATTERS = {unicode: lambda x: x.encode('utf8')}

# The default value for the include_index argument in create
default_include_index = True


def EscapeRawText():
  """Automatically escape any html present for any non-custom formatted data.

  Use Html function below to protect the columns that should be html.
  """
  DEFAULT_FORMATTERS[basestring] = FormatAsText


def FormatAsHtml(x):
  """Use this to annotate your column data as html.

  Example:
    interactive_table.from_data([['<b> html</b>', ...]],
    extractor=lambda x:  html(x[0]), Text(x[1]), ...)
  Args:
    x: html string

  Returns:
    Html object.
  """
  return display.HTML(x)


def FormatAsText(x):
  """Escapes text."""
  return cgi.escape(x)


# Can override these to change default behavior of interactive tables
DEFAULT_MAX_ROWS = 20000
DEFAULT_MAX_COLUMNS = 20
DEFAULT_ROWS_PER_PAGE = 10
DEFAULT_MAX_DATA_SIZE = 5000000

def create(dataframe, include_index=None, **kwargs):
  """Creates interactive table from pandas DataFrame.

  Args:
      dataframe: exposes columns and data (e.g.
        pandas.core.frame.DataFrame)
      include_index: whether to include the index as a column. If None, uses
          default_include_index
      **kwargs: any arguments that can be passed to the constructor of
          L{GVizTable<interactive_table.GVizTable>} verbatim.
  Returns:
      GVizTable wrapping dataframe.
  """
  if include_index is None:
    include_index = default_include_index
  if include_index or dataframe.shape[1] == 0:
    dataframe = dataframe.reset_index()
  if not dataframe.columns.is_unique:
    df_copy = dataframe.copy(deep=False)
    df_copy.columns = range(dataframe.shape[1])
    records = df_copy.to_records(index=False)
  else:
    records = dataframe.to_records(index=False)
  max_columns = kwargs.get('max_columns', DEFAULT_MAX_COLUMNS)
  # Doing the simpler records.tolist() will make a *copy*, but slicing will (as
  # of numpy 1.13) will create a *view*, cf:
  # https://docs.scipy.org/doc/numpy-dev/release.html#multiple-field-manipulation-of-structured-arrays
  data = records[list(records.dtype.names[:max_columns])]
  return GVizTable(columns=dataframe.columns, data=data, **kwargs)


def from_data(data, extractor=lambda x: x, columns=None, **kwargs):
  """Creates interactive table from any given data.

  Args:
    data: an iterable, where each element provides the data needed
        to render a row.
    extractor: function that takes one data element and returns a list
        of elements to display for each row. See various U{formatters.*}
        for examples.
    columns: list of column names (passed verbatim to constructor)
    **kwargs: any arguments to pass to L{GVizTable.__init__}

  Returns:
    L{GVizTable}
  """
  data = [extractor(each) for each in data]
  data = [each for each in data if each is not None]
  if not data:
    columns = []
  if columns is None:
    columns = range(len(data[0]))
  return GVizTable(columns=columns, data=data, **kwargs)


class GVizTable(display.DisplayObject):
  """A GViz Table interactive widget."""

  def __init__(self,
               columns,
               data,
               num_rows_per_page=None,
               max_rows=None,
               max_columns=None,
               max_data_size=None,
               column_widths=None,
               custom_formatters=None,
               publish_immediately=False,
               default_formatters=None):
    """Constructor.

    Args:
       columns: a list of column names e.g. ['salary', 'rank']

       data: list of rows, each row of the same size as columns. E.g.
         [['100', 'engineer'], ['99', 'janitor']]. Each element of data is
         number, string or DateTime.

       num_rows_per_page: display that many rows per page initially.
         uses DEFAULT_ROWS_PER_PAGE if not provided.

       max_rows: if len(data) exceeds this value a warning will be printed
         and the table truncated. Uses DEFAULT_MAX_ROWS if not provided

       max_columns: if len(columns) exceeds this value a warning will be
         printed and truncated. Uses DEFAULT_MAX_COLUMNS if not provided

       max_data_size: if the amount of data exceeds this values all remaining
         rows will be discarded. Uses DEFAULT_MAX_DATA_SIZE if not provided

       column_widths: a map from column number to the width style to apply for
         that column.

       custom_formatters: a map from column indices or names to functions or
         functors that apply custom formatting to the data field. The function
         should accept one element from the corresponding column and return
         properly formatted html. For example to render string in the first
         column as image content, the second as url use and third as I{image
         url}, provide
           custom_formatters={
            0: L{formatters.EncodedImage(height=50)},
            1: L{formatters.Url()},
            'image': L{formatters.ImageUrl()}
           }.

         If formatters are given for the index and name of the same column, the
         index version is selected.

         See L{colabtools.formatters<google3.research.colab.lib.formatters>}
         for more formatters.

       publish_immediately: if True, will publish the table immediately
          (and display.display(table) will have no effect. Otherwise
          the table needs to be created as last element of the output,
          or published via display.display. (such as display.display(table))

       default_formatters: an optional mapping from type to formatting function
          Default will use
          L{DEFAULT_FORMATTERS<interactive_table.DEFAULT_FORMATTERS}> which
          is empty by default. Use interactive_table.EscapeRawText() to set
          default to automatically escape any html tags in table content.
    """
    if default_formatters is None:
      default_formatters = DEFAULT_FORMATTERS
    num_rows_per_page = num_rows_per_page or DEFAULT_ROWS_PER_PAGE
    max_data_size = max_data_size or DEFAULT_MAX_DATA_SIZE

    # Using persistent_id here ensures that table will will be accessible by
    # other cells
    self.id = 'IT_' + str(uuid.uuid4())
    max_columns = max_columns or DEFAULT_MAX_COLUMNS
    self.columns = _interactive_table_helper.TrimColumns(columns, max_columns)
    self.data = _interactive_table_helper.TrimData(
        data, max_rows or DEFAULT_MAX_ROWS, max_columns)

    self.column_widths = column_widths or {}
    self.custom_formatters = _interactive_table_helper.ProcessCustomFormatters(
        custom_formatters, columns)
    self.header_formatters = {}
    default_formatter = _interactive_table_helper.FindFormatter(
        default_formatters)

    for i, _ in enumerate(columns):
      if i not in self.custom_formatters:
        self.custom_formatters[i] = default_formatter
      # We don't apply custom_formatters to headers, but we do apply
      # default ones - this allows us to do string escaping, number conversion
      # using the type of the actual object.
      self.header_formatters[i] = default_formatter

    self.num_rows_per_page = num_rows_per_page
    self.published = publish_immediately

    if publish_immediately:
      self.Publish()

  def Publish(self):
    """Publishes the interactive table at the current position.

    table.Publish() is a shortcut for display.display(table)
    """
    display.display(display.HTML(self._get_html()))

  def _repr_html_(self):
    """Used by frontend to generate the actual table.

    Returns:

    html representation and javascript hooks to generate the table.
    """
    return '' if self.published else self._get_html()

  def _get_html(self):
    # implicit evalution of numpy.array into bool.  bad idea!
    if len(self.data) == 0:  # pylint: disable=g-explicit-length-test
      return 'The table is empty'
    return '''
      <link rel="stylesheet" href="/nbextensions/google.colab/gviz_interactive_table.css">
      <script src="/nbextensions/google.colab/gviz_loader.js"></script>
      <script src="/nbextensions/google.colab/gviz_interactive_table_main.js"></script>
      <div id="{id}"></div>
      <script>{initialization}</script>'''.format(id=self.id, initialization=self._gen_js(self.columns, self.data))

  def _gen_js(self, columns, data):
    """Returns javascript for this table."""
    formatted_data = _interactive_table_helper.FormatData(
        data, DEFAULT_NONUNICODE_FORMATTER, self.custom_formatters)
    column_types = formatted_data['column_types']

    columns_and_types = []
    for i, (column_type, column) in enumerate(zip(column_types, columns)):
      columns_and_types.append((column_type,
                                str(self.header_formatters[i](column))))

    widths = 'null'
    if self.column_widths:
      widths = [self.column_widths.get(i, 'null') for i in range(len(columns))]

    return """
      (() => {
        const data = %(data)s;
        const kernelTag = "%(id)s";
        gvizInteractiveTable.create({
          data,
          saveCallback: () => {},
          elementId: "%(id)s",
          columns: %(columns)s,
          columnWidths: %(column_widths)s,
          rowsPerPage: %(num_rows_per_page)d,
        });
      })();
    //# sourceURL=table_%(id)s
    """ % {
        'id': self.id,
        'num_rows_per_page': self.num_rows_per_page,
        'data': formatted_data['data'],
        'columns': json.dumps(columns_and_types),
        'column_widths': widths,
    }
