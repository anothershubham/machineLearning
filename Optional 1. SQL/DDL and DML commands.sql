-- For creating a new database
create database market_star_schema;

-- For using the created database
use market_star_schema;

-- For creating table with given attributes
create table shipping_mode_dimen (
	Ship_Mode Varchar(25),
    Vehicle_Company varchar(25),
    Toll_Required bool
);

-- For dropping the database and table
drop table shipping_mode_dimen;
drop database market_star_schema;

-- For Alter table
-- 	Adding primary key to the table
alter table shipping_mode_dimen
add constraint primary key (Ship_Mode);

-- DML Commands
-- 	1.	Inserting new Data
insert into shipping_mode_dimen
values
('DELIVERY TRUCK','Ashok Leyland', false),
('REGULAR AIR', 'Air India', false);

insert into shipping_mode_dimen(Ship_Mode, Vehicle_Company, Toll_Required)
values
('DELIVERY TRUCK','Ashok Leyland', false),
('REGULAR AIR', 'Air India', false);

-- 2. Updating the table values
update shipping_mode_dimen
set Toll_Required = false
where Ship_Mode='Delivery Truck';

-- 3. Deleting the table values
delete from shipping_mode_dimen
where Ship_Mode='Regular Air';

-- 4. Selecting/Viewing the table values
select * from shipping_mode_dimen








