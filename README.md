# ReLIFE Forecasting Service API


API for a simplified building thermal simulation and EPC generation.

Required functionality (in EN):
1) Load a dictionary of geometric/thermal building data
2) Get (and load) a dictionary for the associated plant
3) Thermally simulate the building using weather data EPW
4) Export the results in CSV
5) Generate an EPC (energy class) with default input

NOTE IMPORTANT
- This is an example of workflow all the content and API name will be different after the integration of teh pybuilding library.

Local run
------------
1) Create a virtualenv and install the packages:
   pip install fastapi uvicorn "pydantic>=2" pandas python-multipart

2) Run the server:
   uvicorn app:app --reload

3) Open the interactive OpenAPI documentation:
   http://127.0.0.1:8000/docs

Typical Workflow 
---------------
- POST /project -> create a project_id
- PUT /project/{id}/building -> load the building dictionary
- GET /plant/template -> get the plant template
- PUT /project/{id}/plant -> load the plant dictionary
- POST /project/{id}/simulate (multipart con file=EPW) -> run the simulation
- GET /project/{id}/results.csv -> download the results CSV
- GET /project/{id}/epc -> get the EPC based on default input

"""


## Introduction

This is a Python template for a ReLIFE Service API that integrates with Supabase for database operations and storage, and with Keycloak for authentication and authorisation. It provides a foundation for building ReLIFE Services including Supabase user authentication, Keycloak role-based access control, and file storage capabilities.

## Technology Stack

- **Python 3+**: Core programming language
- **FastAPI**: Web framework for building APIs with automatic OpenAPI documentation
- **Uvicorn**: ASGI server for running the FastAPI application
- **Pydantic**: Data validation and settings management using Python type annotations
- **Supabase**: Backend-as-a-Service providing database operations and storage
- **Keycloak**: Identity and access management for authentication and authorization
- **HTTPX**: HTTP client library for making requests
- **Rich**: Terminal output formatting and styling
- **Pytest**: Testing framework with async support

## Configuration

All configuration is driven by environment variables:

| Category     | Variable                 | Description                                       | Default Value                                        |
| ------------ | ------------------------ | ------------------------------------------------- | ---------------------------------------------------- |
| **Server**   | `API_HOST`               | Host address for the API server                   | `0.0.0.0`                                            |
|              | `API_PORT`               | Port for the API server                           | `9090`                                               |
| **Supabase** | `SUPABASE_URL`           | URL of the Supabase instance                      | -                                                    |
|              | `SUPABASE_KEY`           | Service role key with admin privileges            | -                                                    |
| **Keycloak** | `KEYCLOAK_CLIENT_ID`     | Client ID for the application in Keycloak         | -                                                    |
|              | `KEYCLOAK_CLIENT_SECRET` | Client secret for the application in Keycloak     | -                                                    |
|              | `KEYCLOAK_REALM_URL`     | Base URL of the Keycloak realm for authentication | `https://relife-identity.test.ctic.es/realms/relife` |
| **Roles**    | `ADMIN_ROLE_NAME`        | Name of the admin role used for permission checks | `relife_admin`                                       |
| **Storage**  | `BUCKET_NAME`            | Name of the default storage bucket in Supabase    | `default_relife_bucket`                              |

> [!WARNING]
> * The `SUPABASE_KEY` uses the service role key that bypasses Row Level Security (RLS) policies. This should **never** be exposed to clients.
> * `KEYCLOAK_CLIENT_SECRET` is sensitive and should be properly secured in production environments.

## Authentication Integration Validation

This template includes a validation script to test authentication integration with remote Supabase and Keycloak instances. This tool helps you verify your configuration and troubleshoot authentication issues.

### Usage

```bash
uv run validate-supabase --email <your-email> --auth-method <method>
```

### Authentication Methods

| Method            | Description                                                    | Use Case                                    |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------- |
| `supabase`        | Email/password authentication via Supabase                     | Testing direct Supabase user authentication |
| `keycloak-user`   | Username/password via Keycloak (Resource Owner Password Grant) | Testing Keycloak user credentials           |
| `keycloak-client` | Client credentials via Keycloak (Client Credentials Grant)     | Testing service-to-service authentication   |

### Validation Process

The script performs an end-to-end authentication validation:

1. **Authentication**: Authenticate using the specified method and credentials
2. **Server Startup**: Launches a temporary API server instance
3. **Endpoint Verification**: Tests the `/whoami` endpoint with the obtained token
4. **User Information**: Displays authenticated user details and associated roles
5. **Cleanup**: Automatically shuts down the temporary server



