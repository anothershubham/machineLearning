use market_star_schema_all;

select Ord_id, round(Sales) as Rounded_Sales, Customer_Name, 
rank() over (order by Sales desc) as Sale_Amt_Rank
from market_fact_full inner join cust_dimen using (Cust_id);