-- Show and add orders
SELECT * FROM items;
SELECT "--";
SELECT * FROM orders;
SELECT "--";

INSERT INTO orders (item_name, number) VALUES ('apple', 1);
INSERT INTO orders (item_name, number) VALUES ('apple', 3);
INSERT INTO orders (item_name, number) VALUES ('pear', 2);

SELECT "--";

SELECT * FROM items;
SELECT "--";
SELECT * FROM orders;