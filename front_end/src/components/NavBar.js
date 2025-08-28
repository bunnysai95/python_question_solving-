import { useContext, useState } from "react";
import {
  Navbar, Nav, Form, FormControl, Button, Badge, InputGroup, Container
} from "react-bootstrap";
import { Link } from "react-router-dom";
import { ProductContext } from "../ProductContext";

export default function NavBar() {
  const [search, setSearch] = useState("");
  const [products, setProducts] =
    useContext(ProductContext) ?? [{ data: [] }, () => {}];

  const count = Array.isArray(products?.data) ? products.data.length : 0;

  const filterProduct = (e) => {
    e.preventDefault();
    const q = search.trim().toLowerCase();
    if (!q) return;
    const base = Array.isArray(products?.data) ? products.data : [];
    const filtered = base.filter(p =>
      String(p?.name ?? "").toLowerCase().includes(q)
    );
    setProducts({ data: filtered });
  };

  return (
    <Navbar bg="dark" variant="dark" expand="lg" className="mb-3">
      <Container fluid>
        <Navbar.Brand as={Link} to="/">Inventory Management App</Navbar.Brand>

        <Navbar.Toggle aria-controls="topbar" />
        <Navbar.Collapse id="topbar">
          <Nav className="me-auto align-items-center">
            <Badge bg="primary" className="ms-2">
              Products In stock {count}
            </Badge>
          </Nav>

          <Form onSubmit={filterProduct} className="d-flex align-items-center">
            <Button as={Link} to="/addproduct" size="sm" className="me-3">
              Add Product
            </Button>

            <InputGroup className="flex-nowrap">
              <FormControl
                placeholder="Search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="me-2"
                style={{ width: 280 }}
              />
              {/* In v2, put the button directly after FormControl */}
              <Button type="submit" variant="outline-primary">Search</Button>
            </InputGroup>
          </Form>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}
