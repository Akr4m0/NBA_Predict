# Common Warnings and Fixes

## Fixed Warnings

### 1. SettingWithCopyWarning (Pandas)
**Problem**: Direct assignment to DataFrame slices can trigger warnings
```python
df['column'] = values  # Can cause warning
```

**Solution**: Use `.loc[]` for explicit assignment
```python
df.loc[:, 'column'] = values  # Fixed
```

### 2. Sklearn Metric Warnings
**Problem**: Division by zero in precision/recall calculations
```python
precision_score(y_true, y_pred, average='weighted')  # Can warn about zero division
```

**Solution**: Add `zero_division` parameter
```python
precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Fixed
```

### 3. Missing Type Hints
**Problem**: Missing `List` import for type hints
```python
def function(param: List[str]):  # NameError: name 'List' is not defined
```

**Solution**: Import `List` from typing
```python
from typing import List
def function(param: List[str]):  # Fixed
```

## Additional Warning Fixes

### 4. Future Warnings (Pandas)
If you see warnings about deprecated methods:

```python
# Replace deprecated methods:
df.applymap(func)  # Deprecated
df.map(func)       # Use this instead

# For fillna with method parameter:
df.fillna(method='forward')  # Deprecated  
df.fillna(df.shift())        # Use this instead
```

### 5. Numpy Warnings
```python
# Replace deprecated numpy types:
np.int    # Deprecated
np.int64  # Use this

np.float  # Deprecated  
np.float64# Use this
```

### 6. Plotly/Dash Warnings
If you get Plotly warnings, update the import:
```python
# Old way (may cause warnings):
import plotly.plotly as py

# New way:
import plotly.graph_objects as go
import plotly.express as px
```

## Running Without Warnings

To suppress specific warnings temporarily:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
```

To see all warnings:
```bash
python -W all your_script.py
```

## Installation Issues

If you're missing packages, install them:
```bash
pip install -r requirements.txt
```

If specific versions cause conflicts:
```bash
pip install --upgrade pandas scikit-learn plotly dash
```