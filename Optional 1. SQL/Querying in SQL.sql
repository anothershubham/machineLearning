use market_star_schema_all;

show tables;

-- Selecting the table and counting the rows
select * from shipping_dimen;
select count(*) from shipping_dimen;
select * from cust_dimen;
select count(Customer_Name) as 'Customer Name' from cust_dimen;

-- Renaming column names while giving output with and without 'AS' keyword
select Customer_name 'Cust Name', City 'City Name', State 'State Name' from cust_dimen; 

select Customer_name 'Cust Name', City 'City Name', State 'State Name' from cust_dimen;

-- Using WHERE clause
select Customer_name 'Cust Name', City 'City Name', State 'State Name' from cust_dimen 
where State= 'Maharashtra';

-- 	'WHERE' clause with 'AND' clause
select * from cust_dimen 
where State= 'Maharashtra' and Customer_Segment = 'Corporate';

-- 	'WHERE' clause with 'OR' clause
select * from cust_dimen 
where State= 'Maharashtra' or Customer_Segment = 'Corporate';

-- 'IN' clause: Selecting customers from a list of provided state
select * from cust_dimen 
where State in ('Karnataka','Maharashtra');

-- != usage
select * from cust_dimen 
where State != 'Karnataka';

-- Print the orders with losses
select * from market_fact_full
where profit<0;

-- 'LIKE' and between usage: Print orders with '_5' in order nos. and shipping cost between 10 and 15
select * from market_fact_full
where Ord_id like '%\_5%' and Shipping_cost between 10 and 20;

-- display the cities in the cust_dimen table which begin with the letter 'K'
select * from cust_dimen
where City like 'k%';

select count(*) from cust_dimen where Customer_Name > "Joshi" and Customer_Name < "Koshi";
select count(*) from cust_dimen where Customer_Name between "Joshi" and  "Koshi";


select * from market_fact_full
where Market_fact_id between 2 and 5;

select 'A' = char('65');
select ('A')=char(65);
select 'a'< '54';

-- Using 'GROUP BY' aggregation
select * from market_fact_full;

select count(Ord_id), Cust_id, sum(Discount), mode(Profit) from market_fact_full group by Cust_id;

select Cust_id,Prod_id from market_fact_full
group by Cust_id, Prod_id;

-- Using 'Order by' clause
select distinct Cust_id from market_fact_full order by Cust_id desc;

select Prod_id, Product_Category from prod_dimen order by Product_Category, Prod_id;

select Ship_Date, Ship_id from shipping_dimen order by Ship_Date, Ship_id;

-- Using 'limit' clause
select Ship_Date, Ship_id from shipping_dimen order by Ship_Date, Ship_id limit 3;


-- Using 'having' clause
select Prod_id, sum(Order_Quantity) from market_fact_full group by Prod_id having 
sum(Order_Quantity)>10 order by sum(Order_Quantity) limit 3;

-- String functions
select concat(Manu_Id, ' ', Manu_Name, ' ', Manu_City) from manu;

select Customer_Name, concat(upper(substring(substring_index(lower(customer_name),' ',1),1,1)),
upper(substring(substring_index(lower(customer_name),' ',-1),1,1))) as Initials from cust_dimen;


-- Common Table Expression(CTE)
with table1 as (select Customer_Name, concat(upper(substring(substring_index(lower(customer_name),' ',1),1,1)),
upper(substring(substring_index(lower(customer_name),' ',-1),1,1))) as Initials from cust_dimen), 
table2 as (select Cust_id,Prod_id from market_fact_full)
select count(Customer_Name), Initials from table1 cross join table2 group by Customer_Name;

with table1 as (select Customer_Name, concat(upper(substring(substring_index(lower(customer_name),' ',1),1,1)),
upper(substring(substring_index(lower(customer_name),' ',-1),1,1))) as Initials from cust_dimen) select * from table1;
with table2 as (select Cust_id,Prod_id from market_fact_full) select * from table2;
select f.Initials, g.Cust_id from table1 f cross join table2 g;

-- View Clause
create view custom1 as with table1 as (select Customer_Name, concat(upper(substring(substring_index(lower(customer_name),' ',1),1,1)),
upper(substring(substring_index(lower(customer_name),' ',-1),1,1))) as Initials from cust_dimen), 
table2 as (select Cust_id,Prod_id from market_fact_full)
select Customer_Name, Prod_id from table1 cross join table2;

drop view custom1;

select * from custom1;





