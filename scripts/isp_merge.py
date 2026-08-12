
import os
import pandas as pd


if __name__ == "__main__":
    DATA_DIR = "../../Database"
    isp_file = pd.read_csv(os.path.join(DATA_DIR, "ISP Case Data 1.csv"))
    serial_numbers = pd.read_excel(os.path.join(DATA_DIR, "Serial Numbers.xlsx"))

    temp = pd.DataFrame(serial_numbers.iloc[2:]).reset_index(drop=1)
    new_map = {
        old:new
        for old, new in zip(serial_numbers.columns.to_list(), serial_numbers.iloc[1])
    }
    serial_numbers = temp.rename(new_map, axis=1)
    month_dic = dict(zip(pd.date_range('2000-01-01', freq='M', periods=12).strftime('%B'), range(1,13)))
    merge = isp_file.merge(
        serial_numbers,
        left_on="Serial Number",
        right_on="serial_number"
    )
    merge.head()
    merge = merge[["charging_station_id", "Serial Number", "Title", "HTML Description", "Product Family", "Year", "Month", "Day", "country[site_info]"]]
    merge["Date"] = pd.Series([pd.Timestamp(year=y,month=month_dic[m],day=d) for y,m,d in merge[["Year","Month","Day"]].values.tolist()])
    merge.to_csv(os.path.join(DATA_DIR, "Merged_ISP.csv"))