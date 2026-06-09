# Multi-stage build: vite builds static assets, nginx serves them.

FROM node:20-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Backend URL baked into the SPA bundle at build time. The default below is
# only useful for local docker-compose — in a real cloud deploy the browser
# would try to fetch from its OWN localhost:8000 (which won't exist). You MUST
# override with `--build-arg VITE_API_BASE_URL=https://api.example.com` or set
# the `args:` block in your compose/k8s manifest. There is no runtime override.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build


FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
