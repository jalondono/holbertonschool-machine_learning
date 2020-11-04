--  trigger that decreases the quantity of an item after adding a new order.
delimiter //

CREATE TRIGGER update_items
AFTER INSERT
ON orders
FOR EACH ROW
BEGIN
update items SET quantity = quantity - NEW.number
WHERE items.name=NEW.item_name;
END //
delimiter ;