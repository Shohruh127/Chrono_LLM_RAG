#Uzbek XLSX Preprocessor
# Created by: Shohruh127

import pandas as pd
from datetime import datetime

class UzbekXLSXPreprocessor:
    """Preprocesses Uzbek regional data from XLSX files into a Chronos-2 compatible format.
    It handles Cyrillic transliteration, category translation, and data structuring.
    """

    def __init__(self):
        # Stores mapping from generated series IDs to original location/category names
        self.location_mapping = {}
        # Dictionary for translating common Uzbek Cyrillic category names to English
        self.category_translations = {
            'Саноат': 'Industry',
            'Қишлоқ, ўрмон ва балиқчилик хўжалиги': 'Agriculture',
            'Қишлоқ': 'Agriculture',
            'саноат': 'Industry',
            'қишлоқ': 'Agriculture',
            'Ишлаб чиқариш': 'Manufacturing',
            'Савдо': 'Trade'
        }
        # Abbreviations for categories, used in generating series IDs
        self.category_abbrev = {
            'Industry': 'IND',
            'Agriculture': 'AGR',
            'Manufacturing': 'MFG',
            'Trade': 'TRD',
            'Unknown': 'UNK' # Default for untranslated categories
        }

    def transliterate(self, text):
        """Transliterates Uzbek Cyrillic text to Latin (ASCII) characters.
        This helps in creating consistent, searchable IDs and names.
        """
        if pd.isna(text):
            return None

        text = str(text).strip()

        # Transliteration map for common Uzbek Cyrillic characters
        trans_map = {
            'А': 'A', 'а': 'a', 'Б': 'B', 'б': 'b',
            'В': 'V', 'в': 'v', 'Г': 'G', 'г': 'g',
            'Д': 'D', 'д': 'd', 'Е': 'E', 'е': 'e',
            'Ё': 'Yo', 'ё': 'yo', 'Ж': 'J', 'ж': 'j',
            'З': 'Z', 'з': 'z', 'И': 'I', 'и': 'i',
            'Й': 'Y', 'й': 'y', 'К': 'K', 'к': 'k',
            'Л': 'L', 'л': 'l', 'М': 'M', 'м': 'm',
            'Н': 'N', 'н': 'n', 'О': 'O', 'о': 'o',
            'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r',
            'С': 'S', 'с': 's', 'Т': 'T', 'т': 't',
            'У': 'U', 'у': 'u', 'Ф': 'F', 'ф': 'f',
            'Х': 'X', 'х': 'x', 'Ц': 'Ts', 'ц': 'ts',
            'Ч': 'Ch', 'ч': 'ch', 'Ш': 'Sh', 'ш': 'sh',
            'Щ': 'Shch', 'щ': 'shch', 'Ъ': '', 'ъ': '',
            'Ы': 'Y', 'ы': 'y', 'Ь': '', 'ь': '',
            'Э': 'E', 'э': 'e', 'Ю': 'Yu', 'ю': 'yu',
            'Я': 'Ya', 'я': 'ya',
            'Ў': 'O', 'ў': 'o', 'Қ': 'Q', 'қ': 'q',
            'Ғ': 'G', 'ғ': 'g', 'Ҳ': 'H', 'ҳ': 'h'
        }

        result = ''
        for char in text:
            result += trans_map.get(char, char) # Replace if in map, otherwise keep original char

        # Keep only alphanumeric characters, spaces, and dots for cleaned names
        result = ''.join(c for c in result if c.isalnum() or c in ' .')

        return result.strip()

    def translate_category(self, cat_name):
        """Translates Uzbek category names to English based on predefined mappings."""
        if pd.isna(cat_name):
            return 'Unknown'

        cat_str = str(cat_name).strip()

        # Check for exact matches first
        if cat_str in self.category_translations:
            return self.category_translations[cat_str]

        # Check for partial (case-insensitive) matches
        for uzbek, english in self.category_translations.items():
            if uzbek.lower() in cat_str.lower():
                return english

        return 'Unknown' # Return 'Unknown' if no translation is found

    def is_uzbek_regional_format(self, df):
        """Detects if a DataFrame appears to be in the Uzbek regional data format.
        Checks for Cyrillic characters and typical year patterns.
        """
        try:
            # Check for Cyrillic characters in the first few rows (a strong indicator)
            first_rows = df.head(5).to_string()
            has_cyrillic = any(ord(c) > 1000 for c in first_rows) # Cyrillic characters have high Unicode values

            # Check for year patterns in the second row, typical for this dataset format
            second_row = df.iloc[1] if len(df) > 1 else pd.Series()
            has_years = False
            for val in second_row:
                if pd.notna(val):
                    try:
                        year = int(float(val))
                        if 2000 <= year <= 2030: # Look for years in a plausible range
                            has_years = True
                            break
                    except:
                        pass # Ignore non-numeric values

            return has_cyrillic and has_years
        except:
            return False # If any error occurs during detection, assume it's not the format

    def process_uzbek_xlsx(self, file_path):
        """
        Processes an Uzbek regional XLSX file to extract time series data
        and format it for Chronos-2, including transliteration and gap filling.

        Returns:
            (prepared_df, report, mapping_dict): A tuple containing the prepared DataFrame,
            a detailed processing report, and a mapping dictionary for series IDs.
        """

        report = "## 🇺🇿 Uzbek Regional Data Processing\n\n"

        try:
            # Read the Excel file without assuming a header, as the header is multi-row/complex
            df = pd.read_excel(file_path, header=None)

            report += f"**File:** {file_path.split('/')[-1]}\n"
            report += f"**Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns\n\n"

            categories = []
            year_ranges = {}

            # Iterate through columns to find category names (first row) and years (second row)
            for col_idx in range(len(df.columns)):
                # Categories are typically in the first row, are strings, and have some length
                cat_value = df.iloc[0, col_idx]
                if pd.notna(cat_value) and isinstance(cat_value, str) and len(cat_value) > 5:
                    translated = self.translate_category(cat_value)
                    categories.append({
                        'name': translated, # English translated name
                        'original': cat_value.strip(), # Original Uzbek name
                        'start_col': col_idx # Starting column for this category's data
                    })

                # Years are typically in the second row and are numeric
                year_value = df.iloc[1, col_idx]
                if pd.notna(year_value):
                    try:
                        year = int(float(year_value))
                        if 2000 <= year <= 2030: # Validate year range
                            year_ranges[col_idx] = year
                    except:
                        pass # Skip if not a valid year

            if not categories:
                return None, "❌ No categories detected in the file", {} # Error if no categories are found

            report += "### 📋 Detected Categories\n\n"
            for cat in categories:
                report += f"- **{cat['original']}** → {cat['name']}\n"

            report += f"\n### 📅 Years Found\n\n"
            years_sorted = sorted(set(year_ranges.values()))
            report += f"`{years_sorted}`\n\n"

            # Determine the column range for each detected category
            for i in range(len(categories)):
                cat = categories[i]
                start_col = cat['start_col']
                # The end column for a category is the start column of the next category, or the end of the DataFrame
                end_col = categories[i + 1]['start_col'] if i + 1 < len(categories) else len(df.columns)

                # Filter years that fall within the current category's column range
                cat_years = {col: year for col, year in year_ranges.items() if start_col <= col < end_col}
                categories[i]['end_col'] = end_col
                categories[i]['year_columns'] = cat_years

            report += "### 📊 Category Ranges\n\n"
            for cat in categories:
                report += f"- **{cat['name']}**: {len(cat['year_columns'])} years\n"

            # Extract location names, typically from the second column starting from the third row
            location_names = []
            location_orig_map = {}

            for idx in range(2, len(df)): # Start from row 2 (0-indexed) after header rows
                location = df.iloc[idx, 1] # Location name is in the second column

                if pd.isna(location):
                    continue

                location_str = str(location).strip()

                # Skip rows that are clearly headers or summaries within the data body
                skip_keywords = ['hajmli', 'viloyatlar', 'хажмли', 'вилоятлар']
                if any(kw in location_str.lower() for kw in skip_keywords):
                    continue

                # Transliterate the Uzbek location name
                cleaned = self.transliterate(location_str)

                if cleaned and cleaned not in location_names:
                    location_names.append(cleaned)
                    location_orig_map[location_str] = cleaned # Map original to transliterated name

            if not location_names:
                return None, "❌ No locations detected in the file", {} # Error if no locations found

            # Create unique IDs for each location (e.g., LOC_001, LOC_002)
            location_id_map = {name: f"LOC_{i+1:03d}" for i, name in enumerate(sorted(location_names))}

            report += f"\n### 🏷️ Locations Identified: {len(location_id_map)}\n\n"
            for name, loc_id in sorted(location_id_map.items())[:10]: # Show top 10 locations
                report += f"- **{loc_id}**: {name}\n"

            if len(location_id_map) > 10:
                report += f"- ... and **{len(location_id_map) - 10}** more\n"

            # Extract actual data points for each location, category, and year
            all_data = []

            for cat in categories:
                for idx in range(2, len(df)): # Iterate through data rows
                    row = df.iloc[idx]
                    location = row.iloc[1] # Get location name from the second column

                    if pd.isna(location):
                        continue

                    location_str = str(location).strip()

                    # Re-check skip keywords for robustness
                    skip_keywords = ['hajmli', 'viloyatlar', 'хажмли', 'вилоятлар']
                    if any(kw in location_str.lower() for kw in skip_keywords):
                        continue

                    cleaned = location_orig_map.get(location_str)
                    if not cleaned or cleaned not in location_id_map:
                        continue # Skip if location name is not valid or mapped

                    loc_id = location_id_map[cleaned]

                    # Extract values for each year within the current category's columns
                    for col_idx, year in cat['year_columns'].items():
                        value = row.iloc[col_idx]

                        if pd.notna(value):
                            try:
                                # Handle potential comma decimal separators and convert to float
                                value_str = str(value).replace(',', '.')
                                value_float = float(value_str)

                                all_data.append({
                                    'id': loc_id, # Generated location ID
                                    'location_name': cleaned, # Transliterated name
                                    'category': cat['name'], # English category name
                                    'year': year,
                                    'target': value_float # The numerical value
                                })
                            except:
                                continue # Skip if value cannot be converted to float

            if not all_data:
                return None, "❌ No data extracted from the file", {} # Error if no data points are extracted

            # Create a raw DataFrame from extracted data
            df_raw = pd.DataFrame(all_data)
            # Aggregate data by (id, location_name, category, year) to sum up values if multiple entries exist
            df_agg = df_raw.groupby(['id', 'location_name', 'category', 'year'])['target'].sum().reset_index()

            # Create a unique series ID by combining location ID and category abbreviation
            df_agg['category_code'] = df_agg['category'].map(self.category_abbrev)
            df_agg['series_id'] = df_agg['id'] + '_' + df_agg['category_code']
            # Convert year to datetime object (Chronos expects timestamps)
            df_agg['timestamp'] = pd.to_datetime(df_agg['year'].astype(str) + '-01-01')

            # Prepare DataFrame in the specific format required by Chronos-2 ('id', 'timestamp', 'target')
            chronos_df = df_agg[['series_id', 'timestamp', 'target']].copy()
            chronos_df.columns = ['id', 'timestamp', 'target']

            # Fill time series gaps to ensure continuous data for forecasting
            report += f"\n### 🔍 Filling Time Series Gaps\n\n"

            complete_data = []
            gaps_filled = 0

            for series_id in chronos_df['id'].unique():
                series_data = chronos_df[chronos_df['id'] == series_id].copy()
                series_data = series_data.sort_values('timestamp')

                min_year = series_data['timestamp'].dt.year.min()
                max_year = series_data['timestamp'].dt.year.max()

                # Generate a complete date range for the series (yearly start frequency)
                complete_dates = pd.date_range(
                    start=f'{min_year}-01-01',
                    end=f'{max_year}-01-01',
                    freq='YS' # Year Start frequency
                )

                original_count = len(series_data)

                series_data = series_data.set_index('timestamp')
                series_data = series_data.reindex(complete_dates) # Reindex to the complete date range
                series_data['target'] = series_data['target'].interpolate(method='linear') # Linear interpolation for gaps
                series_data['target'] = series_data['target'].ffill().bfill() # Forward and backward fill any remaining NaNs

                series_data['id'] = series_id
                series_data = series_data.reset_index()
                series_data.rename(columns={'index': 'timestamp'}, inplace=True)

                filled = len(series_data) - original_count
                if filled > 0:
                    gaps_filled += filled

                complete_data.append(series_data[['id', 'timestamp', 'target']])

            chronos_df = pd.concat(complete_data, ignore_index=True) # Combine all processed series
            chronos_df = chronos_df.sort_values(['id', 'timestamp']).reset_index(drop=True)

            if gaps_filled > 0:
                report += f"✅ Filled **{gaps_filled}** missing data points via interpolation\n\n"
            else:
                report += f"✅ No gaps detected - data is complete\n\n"

            # Create a comprehensive mapping dictionary for each generated series ID
            mapping_dict = {}
            for series_id in chronos_df['id'].unique():
                # Parse the series_id to get location ID and category code
                parts = series_id.split('_')
                if len(parts) >= 3:
                    loc_id = f"{parts[0]}_{parts[1]}"
                    cat_code = parts[2]

                    # Find the corresponding original location name and full category name
                    matching = df_agg[df_agg['series_id'] == series_id]
                    if not matching.empty:
                        loc_name = matching.iloc[0]['location_name']
                        cat_full = matching.iloc[0]['category']

                        mapping_dict[series_id] = {
                            'location_id': loc_id,
                            'location_name': loc_name,
                            'category_code': cat_code,
                            'category_full': cat_full
                        }

            # Final report summary
            report += f"### ✅ Processing Complete!\n\n"
            report += f"**Total Records:** {len(chronos_df):,}\n"
            report += f"**Time Series:** {chronos_df['id'].nunique()}\n"
            report += f"**Period:** {chronos_df['timestamp'].dt.year.min()}-{chronos_df['timestamp'].dt.year.max()}\n\n"

            report += "### 📊 Sample Time Series\n\n"
            for series_id in sorted(chronos_df['id'].unique())[:5]: # Show first 5 sample series
                series_data = chronos_df[chronos_df['id'] == series_id]
                info = mapping_dict.get(series_id, {})
                years = f"{series_data['timestamp'].dt.year.min()}-{series_data['timestamp'].dt.year.max()}"
                avg = series_data['target'].mean()
                report += f"- **{series_id}**: {info.get('location_name', 'Unknown')} - {info.get('category_full', 'Unknown')} ({years}, avg={avg:.2f})\n"

            if chronos_df['id'].nunique() > 5:
                report += f"- ... and **{chronos_df['id'].nunique() - 5}** more series\n"

            report += f"\n✅ **Data is ready for Chronos-2 forecasting!**\n"

            # Store the generated mapping for later use in RAG system
            self.location_mapping = mapping_dict

            return chronos_df, report, mapping_dict

        except Exception as e:
            import traceback
            error_report = f"❌ **Error Processing Uzbek XLSX**\n\n"
            error_report += f"**Error Type:** {type(e).__name__}\n\n"
            error_report += f"**Error Message:**\n```\n{str(e)}\n```\n\n"
            error_report += f"**Traceback:**\n```\n{traceback.format_exc()}\n```"
            return None, error_report, {}

# Initialize the Uzbek XLSX Preprocessor
uzbek_preprocessor = UzbekXLSXPreprocessor()

print("✅ Uzbek XLSX Preprocessor ready!")
print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: Shohruh127")
