select * from cust_dimen c right join market_fact_full m on c.Cust_id=m.Cust_id;

select * from cust_dimen c left join market_fact_full m on c.Cust_id=m.Cust_id;

select * from cust_dimen c cross join market_fact_full m on c.Cust_id=m.Cust_id;

select * from cust_dimen c cross join market_fact_full m on c.Cust_id=m.Cust_id inner join
shipping_dimen s where s.Ship_id=m.Ship_id;

select m.Ord_id, count(Order_Quantity) from cust_dimen c cross join market_fact_full m on c.Cust_id=m.Cust_id inner join
shipping_dimen s where s.Ship_id=m.Ship_id group by m.Ord_id order by count(Order_Quantity) desc;